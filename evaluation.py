import re
import nltk
import json
import random
from tqdm import tqdm
from nltk.util import ngrams
from inference_class import Inference
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score, accuracy_score


class Metric:
    def __init__(
        self, 
        model_path: str = "",
        json_path: str = "", 
    ) -> None:

        self.val_data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            self.val_data = json.load(f)
        

        self.inference = Inference(model_path)
        
        self.bleu2_score = []
        self.meteor_score = []
        self.rouge_p_score = []
        self.rouge_recall_score = []
        self.rouge_f1_score = []
        self.gt = []
        self.pred = []
        self.overlap_score = []
    
    
    def calculate(self):
        # 计算指标
        for item in tqdm(self.val_data):

            candidate = self.inference.vqa(video_path = "./dataset" + item["path"][1:],
                    question = item["problem"],
                    problem_type = item["problem_type"])
            pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(pattern, item["solution"], re.DOTALL)
            item["solution"] = matches[0] if matches else item["solution"]
            matches = re.findall(pattern, candidate, re.DOTALL)
            candidate = matches[0] if matches else candidate
            if item["problem_type"] == "multiple choice":
                random_choice = ['A', 'B', 'C', 'D']
                random_number = random.randint(0, 3)

                solu = item["solution"] # 原
                solu2 = random_choice[random_number] # 换
                item["solution"] = solu2
                idx_number = ord(solu) - ord('A') # 原idx   换idx: random_number
                opti = item["options"][idx_number]
                item["options"][idx_number] = solu + item["options"][random_number][1:]
                item["options"][random_number] = solu2 + opti[1:]
                # print(item["options"])
                # 更新输入
                item["problem"] += item["options"][0]
                item["problem"] += item["options"][1]
                item["problem"] += item["options"][2]
                item["problem"] += item["options"][3]
            
            print("\n"*2 + "+"*100)
            print(item["problem_type"])
            # print(item["problem"])
            # print(item["options"])
            print("*" + item["solution"] + "*")
            print("-"*15)
            print("*" + candidate + "*")
            print("+"*100)
            
            # 按照类别计算
            if item["problem_type"] == "short answer questions":
                self.bleu2_score.append(self.bleu2(item["solution"], candidate))
                self.meteor_score.append(self.meteor(item["solution"], candidate))
                a, b, c = self.rouge_n(item["solution"], candidate, n=2)
                self.rouge_p_score.append(a)
                self.rouge_recall_score.append(b)
                self.rouge_f1_score.append(c)
                print("\033[92m bleu2_score: ", self.bleu2_score[-1], "\033[0m \n ")
                print("\033[92m meteor_score: ", self.meteor_score[-1], "\033[0m \n ")
                print("\033[92m rouge-2: ", a, b, c, "\033[0m \n ")
            elif item["problem_type"] == "multiple choice":
                self.gt.append(item["solution"])
                self.pred.append(candidate)
            elif item["problem_type"] == "time":
                gt_match = re.search(r'\[(\d+),(\d+)\]', item["solution"])
                gt_start = int(gt_match.group(1))
                gt_end = int(gt_match.group(2))
                match = re.search(r'\[(\d+),(\d+)\]', candidate)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    self.overlap_score.append(self.overlap_coefficient((start, end), (gt_start, gt_end)))
                else:
                    self.overlap_score.append(0.0)
                print("\033[92m overlap_coefficient: ", self.overlap_score[-1], "\033[0m \n ")
            else:
                print("\n error "*10)
        
        mean_bleu2 = sum(self.bleu2_score) / len(self.bleu2_score)
        mean_meteor = sum(self.meteor_score) / len(self.meteor_score)
        mean_rouge_p = sum(self.rouge_p_score) / len(self.rouge_p_score)
        mean_rouge_recall = sum(self.rouge_recall_score) / len(self.rouge_recall_score)
        mean_rouge_f1 = sum(self.rouge_f1_score) / len(self.rouge_f1_score)
        print("\n"*10 + "-"*100)
        print("\033[92m bleu2: ", round(mean_bleu2, 4), "\033[0m \n ")
        print("\033[92m meteor: ", round(mean_meteor, 4), "\033[0m \n ")
        print("\033[92m rouge-2(precision, recall, f1): ", round(mean_rouge_p, 4), round(mean_rouge_recall, 4), round(mean_rouge_f1, 4), "\033[0m \n ")

        print("-"*100)
        print("\033[92m acc: ", round(self.acc(self.gt, self.pred), 4), "\033[0m \n ")
        print("\033[92m f1: ", round(self.f1(self.gt, self.pred), 4), "\033[0m \n ")

        mean_overlap = sum(self.overlap_score) / len(self.overlap_score)
        print("-"*100)
        print("\033[92m Overlap Cofficient: ", round(mean_overlap, 4), "\033[0m \n ")
        

        evaluation_result = {
            "short answer questions": {
                "Bleu2": round(mean_bleu2, 4),
                "Meteor": round(mean_meteor, 4),
                "Rouge-2": {
                    "P": round(mean_rouge_p, 4), 
                    "Recall": round(mean_rouge_recall, 4), 
                    "F1": round(mean_rouge_f1, 4)
                }
            },
            "multiple choice": {
                "Acc": round(self.acc(self.gt, self.pred), 4),
                "F1": round(self.f1(self.gt, self.pred), 4),
            },
            "time": {
                "Overlap Cofficient": round(mean_overlap, 4),
            }
        }
        with open("evaluation_Qwen2-VL-2B-GRPO-Only2.json", "w") as f:
            json.dump(evaluation_result, f, indent=4, ensure_ascii=False)
    

    def bleu2(self, gt_text, candidate):
        gt = [gt_text.split()]
        candidate = candidate.split()
        score = sentence_bleu(gt, candidate, weights=(0.5, 0.5, 0, 0)) # 权重表示1-gram和2-gram的权重，这里计算BLEU-2
        return score
    
    
    # Meteor
    def meteor(self, gt_text, candidate):
        gt = [gt_text.split()]
        predict = candidate.split()
        score = meteor_score(gt, predict)
        return score
    
    
    # Rouge-2
    def rouge_n(self, reference, candidate, n=2):
        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())
        ref_ngrams = list(ngrams(ref_tokens, n))
        cand_ngrams = list(ngrams(cand_tokens, n))
        overlap = len(set(ref_ngrams) & set(cand_ngrams))
        precision = overlap / max(1, len(cand_ngrams))
        recall = overlap / max(1, len(ref_ngrams))
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1
    
    
    # Acc
    def acc(self, gt, pred):
        return accuracy_score(gt, pred)
    
    
    # F1
    def f1(self, gt, pred):
        return f1_score(gt, pred, average='macro')  # 多分类F1
    

    # OC【Overlap Cofficient】
    def overlap_coefficient(self, interval_a, interval_b) -> float:
        a_start, a_end = interval_a
        b_start, b_end = interval_b
        if a_start > a_end or b_start > b_end:
            return 0.0    # 检查区间有效性
        # 计算交集
        intersection_start = max(a_start, b_start)
        intersection_end = min(a_end, b_end)
        intersection = max(0.0, intersection_end - intersection_start)
        # 计算区间长度
        length_a = a_end - a_start
        length_b = b_end - b_start
        # 处理除零情况（至少一个区间长度为0）
        min_length = min(length_a, length_b)
        if min_length == 0:
            return 0.0 if intersection == 0 else 1.0  # 两零长度区间视为完全重叠
        return intersection / min_length
    
    
    def t_IoU(self, interval_a, interval_b) -> float:
        a_start, a_end = interval_a
        b_start, b_end = interval_b

        if a_start > a_end or b_start > b_end:
            return 0.0

        inter_start = max(a_start, b_start)
        inter_end = min(a_end, b_end)
        intersection = max(0.0, inter_end - inter_start)

        length_a = a_end - a_start
        length_b = b_end - b_start

        union = length_a + length_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    
    def tIoU_05(self, interval_a, interval_b) -> int:
        t = self.temporal_iou(interval_a, interval_b)
        return int(t >= 0.5)

    

if __name__ == "__main__":

    model_path = ""
    json_path = ""
    
    # init
    m = Metric(model_path=model_path, json_path=json_path)
    
    # evaluation
    m.calculate()
    

