import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)


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
        self.result = []
    
    
    def calculate(self):

        for item in tqdm(self.val_data):
            example_result = {}

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

                # print(item["options"])
                solu = item["solution"]
                solu2 = random_choice[random_number]
                item["solution"] = solu2
                idx_number = ord(solu) - ord('A')
                opti = item["options"][idx_number]
                item["options"][idx_number] = solu + item["options"][random_number][1:]
                item["options"][random_number] = solu2 + opti[1:]

                item["problem"] += item["options"][0]
                item["problem"] += item["options"][1]
                item["problem"] += item["options"][2]
                item["problem"] += item["options"][3]
            
            pattern = r'<think>(.*?)</think>'
            matches = re.findall(pattern, candidate, re.DOTALL)
            think = matches[0] if matches else candidate
            example_result["think"] = think

            print("\n"*2 + "+"*100)
            print(item["problem_type"])
            example_result["id"] = item["id"]
            example_result["problem_type"] = item["problem_type"]
            example_result["Q"] = item["problem"]
            example_result["options"] = item["options"]
            example_result["A"] = item["solution"]
            example_result["A_candidate"] = candidate

            if item["problem_type"] == "short answer questions":
                self.bleu2_score.append(self.bleu2(item["solution"], candidate))
                self.meteor_score.append(self.meteor(item["solution"], candidate))
                a, b, c = self.rouge_n(item["solution"], candidate, n=2)
                self.rouge_p_score.append(a)
                self.rouge_recall_score.append(b)
                self.rouge_f1_score.append(c)
                # print("\033[92m bleu2_score: ", self.bleu2_score[-1], "\033[0m \n ")
                # print("\033[92m meteor_score: ", self.meteor_score[-1], "\033[0m \n ")
                # print("\033[92m rouge-2: ", a, b, c, "\033[0m \n ")
                example_result["bleu2_score"] = self.bleu2_score[-1]
                example_result["meteor_score"] = self.bleu2_score[-1]
                example_result["rouge-2"] = [ a, b, c]
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
                example_result["overlap_coefficient"] = self.overlap_score[-1]
            else:
                print("\n error "*10)
                
            self.result.append(example_result)
        
    def bleu2(self, gt_text, candidate):
        gt = [gt_text.split()]
        candidate = candidate.split()
        score = sentence_bleu(gt, candidate, weights=(0.5, 0.5, 0, 0))
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
        return f1_score(gt, pred, average='macro')
    
    # OC【Overlap Cofficient】
    def overlap_coefficient(self, interval_a, interval_b) -> float:
        """
        公式:OC = |A ∩ B| / min(|A|, |B|)
        返回:float: [0, 1]之间的重叠系数
        示例:overlap_coefficient((30, 44), (35, 50))
        """
        a_start, a_end = interval_a
        b_start, b_end = interval_b
        if a_start > a_end or b_start > b_end:
            return 0.0    

        intersection_start = max(a_start, b_start)
        intersection_end = min(a_end, b_end)
        intersection = max(0.0, intersection_end - intersection_start)

        length_a = a_end - a_start
        length_b = b_end - b_start
        min_length = min(length_a, length_b)
        if min_length == 0:
            return 0.0 if intersection == 0 else 1.0
        return intersection / min_length
    
    

if __name__ == "__main__":
    # path 

    model_path = ""
    json_path = ""
    
    # init
    m = Metric(model_path=model_path, json_path=json_path)
    
    # evaluation
    m.calculate()
    
    with open("example/result/TAR_R1_result.json", "w") as f:
        json.dump(m.result, f, indent=4, ensure_ascii=False)
    

