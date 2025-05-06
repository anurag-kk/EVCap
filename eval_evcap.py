import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from search import beam_search
import random
import numpy as np
from models.evcap import EVCap
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
import nltk
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from datasets import load_dataset
# from cider.cider import Cider
# from spice.spice import Spice

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(img_input):
    if isinstance(img_input, str):  # If input is a file path
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, Image.Image):  # If input is already an Image object
        img = img_input.convert('RGB')
    else:
        raise ValueError("Input must be a file path or a PIL Image object")
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(img).unsqueeze(0)



def validation_whoops(
    args,
    model, 
    tokenizer,    
) -> None:

    device = args.device
    predicts = []
    bleu_scores = []
    meteor_scores = []
    # cider_scores = []
    # spice_scores = []

    # Initialize CIDEr and SPICE scorers
    # cider_scorer = Cider()
    # spice_scorer = Spice()
    annots = []
    imgs = []
    pred_caps = []
    id = 1
    img_id = 0

    examples = load_dataset('nlphuji/whoops', token='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')
    model.eval()
    for example in examples['test']:
        image_id = example['image_id']
        captions = json.loads(example['crowd_captions'])  # Reference captions
        print('\n')
        print(image_id)
        print('GT: ', captions)
        for cap in captions:
            annots.append({"id": id, "image_id": img_id, "caption": cap})
            id += 1
        
        image = example['image']
        image = preprocess_image(image).to(device)
        
        with torch.cuda.amp.autocast(enabled=True):
            qform_all_proj, atts_qform_all_proj = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list)
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            
            bos = torch.ones([batch_size, 1], device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(
                embeddings=embeddings, 
                tokenizer=tokenizer, 
                beam_width=args.beam_width, 
                model=model.llama_model
            )
            sentence = sentence[0]  # Generated caption
            print('Pred: ', sentence)
        # BLEU score calculation with smoothing
        imgs.append({"id": img_id})
        pred_caps.append({"image_id": img_id, "caption": sentence})

        img_id += 1
        # print("-----------")
        # print(f'caps: {captions}')
        # print('len: ', len(captions))
        # print(f'pred: {sentence}')

        # references_tokenized = [word_tokenize(ref) for ref in captions]
        # hypothesis_tokenized = word_tokenize(sentence)

        # print(f'ref tok: {references_tokenized}')
        # print(f'hyp tok: {hypothesis_tokenized}')
        # bleu_score = nltk.translate.bleu_score.sentence_bleu(
        #     references_tokenized, 
        #     hypothesis_tokenized,
        #     smoothing_function=SmoothingFunction().method1
        # )
        # print("-------------")

        # bleu_scores.append(bleu_score)
        # print(f'BLEU: {bleu_score}')
        
        # METEOR score calculation using nltk's `meteor`
        # references_tokenized = [word_tokenize(cap) for cap in captions]
        # hypothesis_tokenized = word_tokenize(sentence)
        # meteor_val = nltk.translate.meteor(references_tokenized, hypothesis_tokenized)
        # meteor_scores.append(meteor_val)
        # print(f'METEOR: {meteor_val}')
        
        # CIDEr score calculation
        # cider, _ = cider_scorer.compute_score({image_id: captions}, {image_id: [sentence]})
        # cider_scores.append(cider)
        # print(f'CIDEr: {cider}')
        
        # # SPICE score calculation
        # spice, _ = spice_scorer.compute_score({image_id: captions}, {image_id: [sentence]})
        # spice_scores.append(spice)
        # print(f'SPICE: {spice}')
        
        predict = {
            "split": 'valid',
            "image_name": image_id,
            "captions": captions,
            "prediction": sentence,
            # "bleu": bleu_score,
            # "meteor": meteor_val,
            # "cider": cider,
            # "spice": spice
        }
        predicts.append(predict)
    
    # Save predictions to a JSON file
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent=4)

    ground_truth_captions = {"annotations": annots, "images": imgs}

    with open("/kaggle/working/ground_truth.json", "w") as f:
        json.dump(ground_truth_captions, f)

    with open("/kaggle/working/predicted.json", "w") as f:
        json.dump(pred_caps, f)

    coco = COCO("ground_truth.json")        # Load ground truth
    cocoRes = coco.loadRes("predicted.json")  # Load predictions

    cocoEval = COCOEvalCap(coco, cocoRes)   # Initialize the evaluation
    cocoEval.evaluate()
    
    # Print overall average scores
    # avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    # avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    # avg_cider = sum(cider_scores) / len(cider_scores) if cider_scores else 0
    # avg_spice = sum(spice_scores) / len(spice_scores) if spice_scores else 0

    # print(f'Average BLEU score (smoothed): {avg_bleu}')
    # print(f'Average METEOR score: {avg_meteor}')
    # print(f'Average CIDEr score: {avg_cider}')
    # print(f'Average SPICE score: {avg_spice}')



def validation_coco_flickr30k(
    args,
    inpath, 
    model,
    tokenizer, 
) -> None:

    device = args.device
    lst = [1007129816, 1009434119, 101362133, 102617084, 10287332, 1039637574, 1043819504, 1043910339, 1044798682, 1071201387, 107969134, 1080230428, 1082250005, 1089059626, 109260218, 109656696, 1104087374, 110671448, 111497985, 1131056918, 1144865997, 1153704539, 1159381599, 1167908324, 1181708011, 118865844, 121178216, 12252043, 1229536824, 123314995, 1250181412, 1253095131, 1255504166, 1258913059, 1281282435, 1287704027, 1295476404, 129860826, 130063845, 1313869424, 1313961775, 133010954, 1332208215, 1333888922, 1341077576, 1351500610, 136581487, 136693281, 1368082221, 139245992, 1395410911, 1396064003, 1397344877, 1404832008, 1408554531, 14133592, 1414911626, 1417882092, 1433088025, 14559446, 1459582913, 1463072715, 146906547, 1489286545, 150411291, 151970521, 1526325728, 155210731, 157910841, 1579198375, 1579206585, 157955034, 160792599, 16151663, 16437914, 16495609, 164969525, 1659358141, 16626851, 166283675, 1664475761, 1675332284, 1681253990, 1690926854, 17186135, 172092464, 1749702972, 175556963, 1798209205, 179828434, 1801663973, 180209719, 1809758121, 183647966, 185404966, 18638572, 1874530310, 1881494074, 1916798494, 1921102799, 1957683077, 196521598, 198534816, 1989609, 202174004, 202175131, 204150602, 2041867793, 2052202553, 2063399864, 2064046067, 2064792226, 2066271441, 2069110603, 2072331027, 2076906555, 2088705195, 211026975, 21138719, 2115631346, 21166859, 2119380659, 212396217, 2131161531, 2152057198, 2165461920, 2165724565, 2169788484, 217949158, 2180356743, 2188599628, 2190227737, 2205958052, 2208662604, 221421001, 2215837479, 221586389, 2217728745, 2255338013, 2255633616, 2259203920, 226481576, 2268207503, 2275214054, 2282260240, 2283966256, 228947041, 2289762817, 23018702, 230486268, 2313085243, 231935782, 2321764238, 2330536645, 2330765551, 2330843604, 2332986053, 2338791560, 2342035754, 2346401538, 2352452459, 2367139509, 2368266191, 2378544134, 238177432, 2391094555, 2391269207, 2392625002, 2407470303, 2415587549, 2421340833, 2421367904, 2422589651, 2424250856, 2424620984, 2433175169, 2436398074, 2438754748, 244073535, 2441354291, 2441818017, 2445442929, 2447284966, 244910130, 2451114871, 245307666, 246231741, 2465441099, 2466171114, 2468466969, 2469878877, 2470317702, 2470493181, 2472980433, 247618600, 247619370, 247704641, 2484190118, 2497343616, 2504007911, 2504764590, 2506460104, 2508851882, 2511760873, 2511798599, 251586160, 2521788750, 2536772737, 2537596840, 2537755800, 2547291721, 2549933281, 2553089098, 2558911884, 2565302802, 2566294211, 2572712647, 2582413611, 2613889835, 261627889, 263104639, 2658439322, 2661138991, 2665461736, 2666111736, 2671167487, 2671560649, 2673209105, 2678668581, 2679382388, 2686849571, 2689001252, 2695961935, 2696866120, 270263570, 2709044515, 2710027673, 271396631, 2714703706, 271572671, 272045297, 2722957422, 2724485630, 2725508159, 2728583298, 2729685399, 273603643, 273699639, 2737609659, 2738077433, 2750185692, 2751602672, 275175200, 2755362721, 2760716468, 2762599124, 2763465196, 2768933636, 277119391, 2773474615, 2773744784, 2778313163, 2780179669, 2789350645, 2795866891, 2797135460, 280007961, 2802158972, 2809218945, 2812568040, 2816259113, 2819466992, 2825327005, 2830561413, 2832487464, 2833502675, 2837640996, 2838888385, 2844641033, 2844747252, 2847514745, 2854959952, 2860040276, 286084055, 2863539038, 2867026654, 2867460039, 2869439070, 2870426310, 2872099696, 2873252292, 2889845164, 2891943949, 2891961886, 2894634533, 289583589, 289638061, 2898810636, 2900242501, 2900560501, 2902844125, 2904997007, 2923386532, 2924306387, 2924489177, 2925163942, 2926595608, 2933637854, 2934022873, 293575531, 2936693681, 2937461473, 2943557287, 2953861572, 2957682838, 2961247549, 2963672852, 2975627633, 2975845158, 2978409165, 2978735290, 2987121689, 2989764675, 2990471798, 299303069, 2993318965, 3005124440, 300577375, 3006095077, 3009047603, 3016244153, 3021953768, 302289651, 3023339840, 3028145992, 3031792444, 3039095384, 3039200576, 3040999637, 3050114829, 3051972592, 3052436578, 3064443326, 3064716525, 3070836317, 3072673694, 3078844565, 3079340229, 3083847439, 3083913737, 3084001782, 3086526292, 3091786541, 3099091086, 3106782647, 3107059919, 3128856481, 313385842, 3135317718, 3138504165, 314739483, 3148193539, 3149894951, 3150380412, 3155400369, 3155657768, 3167453543, 3168354472, 3171020648, 317383917, 3178300150, 3182490771, 3182495095, 319185571, 3192267612, 319750026, 3206999917, 3222702477, 3223318401, 3224375029, 322516292, 3227675485, 3231575742, 3232994074, 3240570205, 3243094580, 3243653344, 3245460937, 3246281818, 3250952067, 3256456935, 3257107194, 3259992164, 3263182360, 3269841412, 327142149, 3274879561, 3277162496, 3278581900, 327955368, 3283368342, 3285342629, 3286543624, 3288174272, 3293751640, 3294830188, 3298457064, 330325191, 3324049172, 3329961466, 3332883133, 3335965982, 3339751521, 3341077091, 3344531479, 3348384389, 3350002347, 3350177228, 3352135319, 3354474353, 3364114507, 3368671163, 3373481779, 3376227992, 3387661249, 3391716619, 3400385314, 3405279045, 3418504074, 3421480658, 3424605029, 3425756814, 3430607596, 3432586199, 3432637363, 3441959314, 3443404125, 3446941415, 3449846784, 3456488632, 3467073304, 3473534758, 3474406285, 3474908743, 3478547247, 3479233432, 3483071906, 3487979741, 3490867290, 3495349745, 3500218173, 3501083764, 3504379699, 3512747808, 3514194772, 3514278386, 3514685438, 3525403875, 3527184455, 3530687486, 3532476966, 3532598048, 3539612893, 354017707, 3541205002, 3543294190, 35437191, 35477421, 3554391279, 3558267192, 3560125106, 3563617591, 3564489441, 3565598162, 3571193625, 3572346664, 3583105294, 3587449716, 3589267801, 3591283677, 3593392955, 3597007663, 3599392711, 3600229533, 3601491447, 3608661756, 3612485097, 3613264553, 3618908551, 3631530817, 363709975, 3638783842, 3642604708, 3643021980, 3646927481, 3647826834, 3660516586, 3665179773, 366548880, 3671851846, 367400736, 3687062281, 3688797852, 3690358949, 3692072577, 3694555931, 3700739661, 3701291852, 3712574653, 3715669736, 3716244806, 3717309680, 3721082512, 3721102409, 3722006169, 3722504168, 3724738804, 3728256505, 3739742770, 374124237, 3773310720, 3780768589, 378434971, 378969774, 3790208395, 3799701220, 3799847884, 3815960082, 3826425403, 3827298104, 3827317206, 384806138, 38541144, 3858873745, 3867804763, 3884010975, 3887428186, 388837010, 3903017514, 3913884179, 3914087743, 3914751903, 3927465948, 3928395936, 3930187102, 39341489, 3938498023, 394707439, 395125320, 396360611, 3966071063, 3969232921, 3970114165, 3974197857, 3996949550, 3999246475, 3999247601, 400598822, 4031513473, 4035354150, 4039846249, 4043318461, 4046112444, 405556084, 4062550863, 4065328632, 407008823, 4075239348, 407569668, 4077122969, 407815946, 4079016275, 4089335666, 4089787993, 409327234, 4093460759, 4095309334, 4103236953, 411175971, 411216802, 4139974800, 4146886427, 4148908583, 4150353377, 4153147942, 415755815, 4158826243, 4161829222, 416992999, 4180952377, 4200930361, 4202061400, 42348693, 4256162530, 4257593776, 4282691555, 4282819676, 428485639, 4284980126, 428979011, 430173345, 4308077016, 43244430, 4337180031, 435054077, 4351734575, 4359872414, 4373983146, 441212506, 441817653, 4418969015, 4420174290, 4427860005, 442983801, 4434125934, 4434578571, 4439439741, 4442323516, 4443087396, 4448853264, 4450153946, 4450821292, 4459992117, 4460747081, 446138054, 4463538995, 4464247181, 4469735380, 4475663002, 4476827011, 4479738973, 4481348193, 4483766693, 4484549063, 4489839598, 449236667, 4494095559, 4495033915, 450697885, 4507759934, 4510809964, 4516267602, 4517193541, 4517838680, 4520820052, 4523132391, 4525077213, 4525821347, 4528578001, 4539608494, 4541692312, 4546499100, 4549977232, 4553348746, 45543081, 4560420776, 4562512283, 4563143284, 4567311889, 4567455846, 4578856261, 4587060991, 4587837067, 4587901777, 4592269543, 4603095253, 4604969760, 4610973875, 4612060755, 4613268345, 4615770260, 4620293662, 462879289, 4630824427, 4631909374, 4634848106, 463551598, 4637947642, 4637951374, 4639459528, 464340358, 4650623132, 4653258614, 4654284177, 4655102365, 4661097013, 4664359066, 4665413015, 4671795847, 4675063973, 4679111230, 4683565867, 4687557453, 4688197510, 4688948552, 4689169924, 4691655601, 4692834620, 4696109052, 4700788144, 4703377742, 4706166634, 4707189762, 4709819574, 4717627685, 4725026501, 4727540499, 4729526023, 4730076543, 4731694958, 4732745499, 4733026480, 4735200580, 4736841029, 4739632460, 4742299638, 4745356451, 474581065, 4751250311, 4752074240, 4752482394, 4752799475, 47529535, 4756010841, 4756089619, 4756089627, 475816542, 4758483073, 4759256692, 4762011238, 4762194732, 476760133, 476769369, 477204750, 4773842539, 47770444, 4780620826, 4786476156, 4787038693, 4788967636, 4789309483, 4789959232, 4795824646, 4796827555, 479807465, 4798837062, 4798986110, 4800006797, 480048562, 4805078127, 4805425261, 4806000438, 4808256003, 4808278005, 4808471657, 4812170955, 4812991208, 4813957025, 4814332291, 4814335297, 4814603619, 4814933116, 481632457, 4817447781, 4818429638, 4821487527, 4823948097, 4824522315, 4826547083, 483039719, 4830409466, 4830651041, 4845544942, 4846324908, 4850814517, 4854547386, 4858175898, 4859764297, 4860086271, 4862041366, 4862204000, 4864584935, 4868221344, 4869914617, 4871230195, 4881733837, 4882632874, 4885361477, 4889181219, 4890769146, 4892698507, 4896595765, 489865145, 4899074189, 4906688033, 4910374312, 4911020894, 4914029409, 4914732942, 4915716087, 4923272678, 4926882194, 4928592495, 4930533574, 4931239366, 4931319718, 4931839897, 4932279873, 4937639267, 4948635454, 4950715878, 4952001654, 4952694407, 4953536921, 4954827712, 4965629392, 4971580248, 4972129585, 4979311570, 4980430291, 4983587808, 499340051, 502529086, 5026046208, 5060850038, 508793006, 509123893, 5094295894, 5103930077, 51145626, 511643051, 5122705505, 5126446040, 514222285, 5163992452, 5169300296, 5222104909, 5225675783, 5229996710, 5238681042, 5244864049, 5246144625, 529101401, 530507000, 5323049335, 532396029, 533508800, 5345473274, 535020523, 5350403659, 5365075444, 5369771639, 538271607, 538825260, 539676201, 5428390334, 54817316, 5489602545, 5491874786, 5498941021, 5501939468, 5506383509, 5506399373, 5510073103, 5519356488, 5522182662, 5566972, 5584269779, 5615068475, 5622966650, 562588230, 5646792433, 566945636, 567903453, 5730226613, 5745120660, 5747873823, 5787072819, 5791274887, 581630745, 5823310445, 5829317322, 5829827359, 583865081, 58579865, 58803866, 5931115247, 5984974054, 5986222648, 5995817000, 6041486114, 6056788662, 6099093979, 6147735412, 616564448, 6224066807, 624080960, 6250527395, 6275000713, 6278649113, 6312211170, 6317293855, 6320815265, 6337111139, 6338542128, 6439261679, 6453399365, 6474645169, 6486182665, 6489895665, 6502187283, 6503917545, 651277216, 6514004309, 6515331737, 6556870225, 6563291133, 6586954247, 6589292543, 6693936663, 6699112713, 6775387932, 6818129694, 6848969863, 6853360659, 6887015851, 6889322961, 6897514777, 6917718093, 6918264972, 6927762908, 6956989138, 697490420, 6978881720, 6995905943, 6999596517, 7001949951, 7003919692, 702083815, 7034076961, 7037413229, 7052724381, 7072093125, 7099368287, 7130336193, 7131011211, 7151070637, 7187734520, 7194590496, 72008434, 7249763658, 7292785488, 7308070356, 7329031116, 7330749240, 7333716086, 733965014, 7348289414, 7393977570, 7438195398, 7446693604, 7468663062, 7544009146, 7559183044, 756521713, 7567712136, 76485985, 766061382, 7670346004, 7694340978, 771048251, 7738684358, 77587237, 7764093618, 783994497, 7890007278, 7900347098, 7988586396, 8016751611, 8052387530, 8052397902, 8132535710, 8183107966, 821071719, 8220955383, 8234387593, 829798661, 83292701, 83493458, 85519095, 86120845, 861661418, 862560775, 86256551, 86350713, 900144365, 94024624, 95758790, 97233789, 97234558]
    with open(inpath, 'r') as infile:
        annotations = json.load(infile)
    # bleu_scores = []
    # meteor_scores = []
    annots = []
    imgs = []
    pred_caps = []
    id = 1
    img_id = 0
    predicts = []
    for i in lst:
        image_id = str(i) + ".jpg"
        captions = annotations[str(i) + ".jpg"]
        image_path = args.image_folder + '/' + image_id
        print('\n')
        print(image_path)
        print('GT: ', captions)
        for cap in captions['comments']:
            annots.append({"id": id, "image_id": img_id, "caption": cap})
            id += 1
        image = preprocess_image(image_path).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            qform_all_proj, atts_qform_all_proj  = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model) # List[str]
            sentence = sentence[0]
            print('Pred: ', sentence)

        imgs.append({"id": img_id})
        pred_caps.append({"image_id": img_id, "caption": sentence})

        img_id += 1

        # references_tokenized = [word_tokenize(ref) for ref in captions['comments']]
        # hypothesis_tokenized = word_tokenize(sentence)

        # print(f'ref tok: {references_tokenized}')
        # print(f'hyp tok: {hypothesis_tokenized}')
        # bleu_score = nltk.translate.bleu_score.sentence_bleu(
        #     references_tokenized, 
        #     hypothesis_tokenized,
        #     smoothing_function=SmoothingFunction().method1
        # )

        # bleu_scores.append(bleu_score)
        # print(f'BLEU: {bleu_score}')
        
        # METEOR score calculation using nltk's `meteor`
        # references_tokenized = [word_tokenize(cap) for cap in captions['comments']]
        # hypothesis_tokenized = word_tokenize(sentence)
        # meteor_val = nltk.translate.meteor(references_tokenized, hypothesis_tokenized)
        # meteor_scores.append(meteor_val)
        # print(f'METEOR: {meteor_val}')
  
        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)

    ground_truth_captions = {"annotations": annots, "images": imgs}

    with open("/kaggle/working/ground_truth.json", "w") as f:
        json.dump(ground_truth_captions, f)

    with open("/kaggle/working/predicted.json", "w") as f:
        json.dump(pred_caps, f)

    coco = COCO("ground_truth.json")        # Load ground truth
    cocoRes = coco.loadRes("predicted.json")  # Load predictions

    cocoEval = COCOEvalCap(coco, cocoRes)   # Initialize the evaluation
    cocoEval.evaluate()                    # Perform evaluation

    # Step 4: Print Scores
    for metric, score in cocoEval.eval.items():
        print(f"{metric}: {score:.3f}")

    # avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    # avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    # print(f'Average BLEU score (smoothed): {avg_bleu}')
    # print(f'Average METEOR score: {avg_meteor}')

def validation_nocaps(
    args,
    inpath,
    model,        
    tokenizer,            
) -> None:
    device = args.device
    with open(inpath, 'r') as infile:
        annotations = json.load(infile)
    indomain = []
    neardomain = []
    outdomain = []
    overall = []
    img_info = json.load(open('/home/nlab/li/research/3_NOC/ours_blip/M_MiniGPT-4/data/nocaps/nocaps_val.json','r'))
    model.eval()
    for idx, annotation in tqdm(enumerate(annotations)):
        ann = img_info[idx]
        image_file = ann['image']
        img_id = ann['img_id']
        image_id = annotation['image_id']
        split = annotation['split']
        captions = annotation['caption']
        print('\n')
        image_path = args.image_folder + '/' + image_file
        print(image_path)
        print('GT: ', captions)
        image = preprocess_image(image_path).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            qform_all_proj, atts_qform_all_proj  = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)

            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)

            sentence_ = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model)
            sentence_ = sentence_[0]
            sentence = sentence_.split('#')[0]
            print('Pred: ', sentence)

            predict = {}
            predict["split"] = split
            predict["image_name"] = image_id
            predict["captions"] = captions
            predict["prediction"] = sentence

            overall.append(predict)
            if split == 'in_domain':
                indomain.append(predict)
            elif split == 'near_domain':
                neardomain.append(predict)
            elif split == 'out_domain':
                outdomain.append(predict)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    with open(os.path.join(args.out_path, f'overall_generated_captions.json'), 'w') as outfile:
        json.dump(overall, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'indomain_generated_captions.json'), 'w') as outfile:
        json.dump(indomain, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'neardomain_generated_captions.json'), 'w') as outfile:
        json.dump(neardomain, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'outdomain_generated_captions.json'), 'w') as outfile:
        json.dump(outdomain, outfile, indent = 4)


@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    print("device: ", device)
    # loading model
    model_type = "lmsys/vicuna-7b-v1.3"
    ckpt = '/kaggle/input/checkpoint/weights.pt'
    print('load:', ckpt)
    model = EVCap(
            ext_path = '/kaggle/input/mydataset/EVCap-main/ext_data/ext_memory_lvis.pkl',
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn = args.topn,
            llama_model=model_type,
            prompt_path="/kaggle/input/mydataset/EVCap-main/prompts/prompt_evcap.txt",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=128,
            end_sym='\n',
            low_resource=True,
            device_8bit=0,
    )
    print("device: ", device)
    state_dict = torch.load(ckpt, map_location=device)['model']


    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    inpath = args.path_of_val_datasets
    tokenizer = model.llama_tokenizer
    if args.name_of_datasets == "nocaps":
        validation_nocaps(args, inpath, model, tokenizer)
    if args.name_of_datasets == "coco" or args.name_of_datasets == "flickr30k":
        validation_coco_flickr30k(args, inpath, model, tokenizer)
    if args.name_of_datasets == "whoops":
        validation_whoops(args, model, tokenizer)


if __name__ == '__main__':
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--name_of_datasets', default = 'whoops', choices = ('coco', 'flickr30k', 'nocaps', 'whoops'))
    parser.add_argument('--path_of_val_datasets', default = '/kaggle/input/flickr30k/Flickr30k/annotations.json')
    parser.add_argument('--image_folder', default = '/kaggle/input/flickr30k/Flickr30k/images')
    parser.add_argument('--out_path', default = '/kaggle/working/generated_captions.json')
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 9)
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    args = parser.parse_args()
    set_seed(args.random_seed)
    print('args: {}\n'.format(vars(args)))
    main(args)
    