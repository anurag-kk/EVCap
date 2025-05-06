import logging
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import random
from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import pickle
import faiss
import re
import torch.nn.functional as F
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import SAGEConv, global_mean_pool

class EVCap(Blip2Base):
    
    def __init__(
        self,
        ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=160,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        print('topn:', self.topn)

        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased").eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # simple graph encoder
        class SimpleGraphEncoder(torch.nn.Module):
            def __init__(self, region_dim, text_dim, hidden_dim, out_dim):
                super().__init__()
                self.fc_region = torch.nn.Linear(region_dim, hidden_dim)
                self.fc_text   = torch.nn.Linear(text_dim,   hidden_dim)
                self.gnn1      = SAGEConv(hidden_dim, hidden_dim)
                self.gnn2      = SAGEConv(hidden_dim, out_dim)

            def forward(self, region_embeds, text_embeds, edge_index, batch):
                x = self.fc_region(region_embeds) + self.fc_text(text_embeds)
                x = F.relu(self.gnn1(x, edge_index))
                x = self.gnn2(x, edge_index)
                return global_mean_pool(x, batch)  # [batch_size, out_dim]

        self.graph_encoder = SimpleGraphEncoder(
            region_dim=768, text_dim=768, hidden_dim=512, out_dim=768
        ).eval()
        for p in self.graph_encoder.parameters():
            p.requires_grad = False

        ##### Image 
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')


        ##### Text 
        self.bert_tokenizer = self.init_tokenizer()
        self.Qformer_txt, self.query_tokens_txt = self.init_Qformer_txt(
            num_query_token_txt, self.Qformer.config.hidden_size
        )
        self.Qformer_txt.resize_token_embeddings(len(self.bert_tokenizer))
        self.Qformer_txt.cls = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer_txt.named_parameters():
                param.requires_grad = False
            self.Qformer_txt = self.Qformer_txt.eval()
            self.Qformer_txt.train = disabled_train
            self.query_tokens_txt.requires_grad = True
            logging.info("freeze Qformer")
        print('query_tokens_txt', self.query_tokens_txt.shape)
        print('Loading Q-Former Done')
        print('Loading Q-Former_txt Done')


        ##### Caption generation 
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        # Load a pre-trained Faster R-CNN model with RPN
        self.fcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.fcnn.eval()  # Set the model to evaluation mode

        ###
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        print(ext_path)
        with open(ext_path, 'rb') as f:
            ext_base_img, self.ext_base_img_id = pickle.load(f)
            print(ext_base_img.shape, len(self.ext_base_img_id))
            feature_library_cpu = ext_base_img.cpu().numpy()
            faiss.normalize_L2(feature_library_cpu)
            self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
            self.feat_index.add(feature_library_cpu)
            print(f"loaded external base image")


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()


    def prompt_wrap(self, img_embeds, atts_img, prompt_list):
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<ImageHere>", 1)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)        
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
                img_embeds_i = img_embeds[i].unsqueeze(0)
                wrapped_embed_i = torch.cat([p_before_embeds, img_embeds_i, p_after_embeds], dim=1)
                emb_lists.append(wrapped_embed_i)  

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img


    def pre_name(self, caption):
        caption = re.sub(
            r"([_!,'\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    def retrieve_similar_features(self, query_features, feat_index, image_id, top_k=1, sub_top_k=1):
        """
        Retrieve one object name per region using the average of the 32 embeddings as the query.

        Args:
            query_features (torch.Tensor): Query features of shape (batch_size, 32, dims).
            feat_index (faiss.Index): FAISS index for similarity search.
            image_id (list): List of object names corresponding to the database embeddings.
            top_k (int): Number of top results to retrieve per query (default: 1).
            sub_top_k (int): Number of top results to return after sorting (default: 1).

        Returns:
            list: List of retrieved object names, one per region.
        """
        batch_size, num_embeddings, dims = query_features.shape

        # Step 1: Average the 32 embeddings for each region
        query_features_avg = query_features.mean(dim=1)  # Shape: (batch_size, dims)

        # Step 2: Convert averaged embeddings to numpy for FAISS
        query_features_avg_cpu = query_features_avg.detach().cpu().numpy()
        faiss.normalize_L2(query_features_avg_cpu)

        # Step 3: Perform similarity search using FAISS
        top_k_similarities, top_k_indices = feat_index.search(query_features_avg_cpu, top_k)

        # Step 4: Convert results to PyTorch tensors
        top_k_indices = torch.tensor(top_k_indices).to(device=query_features.device)
        top_k_similarities = torch.tensor(top_k_similarities).to(device=query_features.device)

        # Step 5: Retrieve object names for the top-k indices
        re_txt_list_all = []
        for batch_i in range(batch_size):
            indices_list = top_k_indices[batch_i]
            re_txt_batch_list = []
            for i in indices_list:
                re_txt_batch_list.append(image_id[i])
            re_txt_list_all.append(re_txt_batch_list)

        # Step 6: Sort and select the top result
        sorted_batched_ret = []
        for listA, listB in zip(top_k_similarities, re_txt_list_all):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [self.pre_name(listB[idx]) for idx in indices]
            sorted_listB = sorted_listB[:sub_top_k]  # Select the top result
            sorted_batched_ret.append(sorted_listB[0])  # Only one result per region

        return sorted_batched_ret
    
    def resize_with_padding(self, imgs, target_size=(224, 224)):
        # Ensure imgs is [B, C, H, W]
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)  # [1, C, H, W]
        elif imgs.ndim != 4:
            raise ValueError(f"Expected shape [B, C, H, W], got {imgs.shape}")

        resized_imgs = []
        for img in imgs:
            _, h, w = img.shape
            scale = min(target_size[0] / h, target_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)

            img_resized = F.interpolate(
                img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)  # [C, new_H, new_W]

            pad_h = target_size[0] - new_h
            pad_w = target_size[1] - new_w
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

            img_padded = F.pad(img_resized, padding, value=0)
            resized_imgs.append(img_padded)

        return torch.stack(resized_imgs)  # [B, C, 224, 224]


    def encode_img(self, images):
        device = images.device
        B = images.size(0)

        print(f"[START] Processing batch of {B} images.")

        # ——— STEP A: Region-level processing & retrieval ———
        all_region_feats = []
        object_texts_per_image = []

        with torch.no_grad():
            self.fcnn.eval()
            for param in self.fcnn.parameters():
                param.requires_grad = False
            for i in range(B):
                print(f"\n[IMAGE {i}]")
                img = images[i].unsqueeze(0)

                # 1⃣ Proposals
                preds = self.fcnn(img)[0]
                boxes, scores = preds['boxes'], preds['scores']
                topk = min(10, scores.size(0))
                topk_scores, topk_indices = scores.topk(topk)
                boxes = boxes[topk_indices]
                keep = scores > 0.1
                boxes = boxes[keep]

                # 2⃣ Crop & resize
                regions = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = map(int, box.tolist())
                    crop = img[0,:,ymin:ymax,xmin:xmax]
                    pad = self.resize_with_padding(crop)
                    regions.append(pad[0])
                if not regions:
                    fallback = F.interpolate(img, size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
                    regions = [fallback]
                region_batch = torch.stack(regions).to(device)

                # 3⃣ Region visual + Q-Former
                if self.low_resource:
                    self.vit_to_cpu()
                    region_batch = region_batch.to('cpu')
                with self.maybe_autocast():
                    img_emb = self.ln_vision(self.visual_encoder(region_batch)).to(device)  # [Ni, seq, D]
                attn = torch.ones(img_emb.size()[:-1], dtype=torch.long, device=device)
                qtoks = self.query_tokens.expand(img_emb.size(0), -1, -1)
                out = self.Qformer.bert(
                    query_embeds=qtoks,
                    encoder_hidden_states=img_emb,
                    encoder_attention_mask=attn,
                    return_dict=True
                )
                toks = out.last_hidden_state  # [Ni, qlen, D]
                region_feats = toks.mean(dim=1)  # [Ni, D]

                # 4⃣ Retrieve names
                names_per_region = self.retrieve_similar_features(toks, self.feat_index, self.ext_base_img_id)
                object_texts_per_image.append(names_per_region)

                all_region_feats.append(region_feats)

        # ——— STEP B: Whole-image -> Q-Former ———
        print("\n[WHOLE IMAGE]")
        if self.low_resource:
            self.vit_to_cpu()
            images = images.to('cpu')
        with self.maybe_autocast():
            whole_embs = self.ln_vision(self.visual_encoder(images)).to(device)
        whole_atts = torch.ones(whole_embs.size()[:-1], dtype=torch.long, device=device)
        qtokens_img = self.query_tokens.expand(B, -1, -1)
        q_out_img = self.Qformer.bert(
            query_embeds=qtokens_img,
            encoder_hidden_states=whole_embs,
            encoder_attention_mask=whole_atts,
            return_dict=True
        ).last_hidden_state
        print(f"  ✅ Whole-image Q-Former out: {q_out_img.shape}")

        # ——— STEP C: GNN ———
        gnn_outputs = []
        for i, region_feats in enumerate(all_region_feats):
            names_per_region = object_texts_per_image[i]
            M = len(names_per_region)
            print(f"\n[GRAPH {i}] Nodes: {M}")
            # text embed per region
            region_sentences = [" ".join(names) for names in names_per_region]
            tokenized = self.bert_tokenizer(region_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                txt_feats = self.text_encoder(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    return_dict=True
                ).last_hidden_state[:,0,:]
            idx = torch.arange(M, device=device)
            if M > 1:
                e = torch.combinations(idx, r=2).t()
                edge_index = torch.cat([e, e.flip(0)], dim=1)
            else:
                edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
            data = Data(region_embeds=region_feats, text_embeds=txt_feats, edge_index=edge_index,
                        batch=torch.zeros(M, dtype=torch.long, device=device))
            gnn_outputs.append(self.graph_encoder(data.region_embeds, data.text_embeds,
                                                data.edge_index, data.batch))
        gnn_feats = torch.cat(gnn_outputs, dim=0)
        print(f"\n✅ GNN output shape: {gnn_feats.shape}")

        # ——— STEP D: Fusion & project ———
        qtokens_txt = self.query_tokens_txt.expand(B, -1, -1)

        # define text-only input IDs and masks
        txt_q_len = qtokens_txt.size(1)
        dummy_ids = torch.zeros((B, txt_q_len), dtype=torch.long, device=device)
        txt_atts = torch.ones((B, txt_q_len), dtype=torch.long, device=device)

        # build encoder states: GNN + full image Q-Former tokens
        img_q_len = q_out_img.size(1)
        encoder_states = torch.cat([
            gnn_feats.unsqueeze(1),    # [B,1,D]
            q_out_img                  # [B,img_q_len,D]
        ], dim=1)  # [B,1+img_q_len,D]

        # build encoder attention mask
        enc_atts = torch.cat([
            torch.ones((B, 1), dtype=torch.long, device=device),
            torch.ones((B, img_q_len), dtype=torch.long, device=device)
        ], dim=1)  # [B,1+img_q_len]

        # run QFormer_txt as decoder: cross-attend to encoder_states
        txt_out = self.Qformer_txt.bert(
            input_ids=dummy_ids,
            attention_mask=txt_atts,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=enc_atts,
            return_dict=True
        ).last_hidden_state  # [B, txt_q_len, D]

        # concatenate image and text tokens
        final_seq = torch.cat([q_out_img, txt_out], dim=1)
        print(f"✅ Sequence before projection: {final_seq.shape}")

        qform_all_proj = self.llama_proj(final_seq)
        atts_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long, device=device)
        return qform_all_proj, atts_proj



    def forward(self, samples):
        ##### Image
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(image)
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(qform_all_proj, atts_qform_all_proj, self.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):

        ##### Caption generation
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)


        bos = torch.ones([qform_all_proj.shape[0], 1],
                         dtype=text_tokens.input_ids.dtype,
                         device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]


        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], 
                       dtype=torch.long).to(image.device).fill_(-100)  
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        
        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_prompt, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"output": outputs[0], "loss": loss}
