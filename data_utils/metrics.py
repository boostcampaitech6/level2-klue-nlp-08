import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

def klue_re_micro_f1(preds, labels):
    """
    KLUE 관계 추출 작업에 대한 micro 평균 F1 score를 계산합니다.
    
    Args:
        preds (list): 예측된 라벨 목록입니다.
        labels (list): 정답 라벨 목록입니다.
    
    Returns:
        float: "no_relation"을 제외한 라벨들에 대한 마이크로 평균 F1 점수(백분율)입니다.
    """
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

# KLUE-RE task의 정밀도-재현율 곡선 아래 영역(AUPRC)을 계산한다.
def klue_re_auprc(probs, labels):
    """
    KLUE 관계 추출 작업에 대한 Precision-Recall Curve 아래 면적(AUPRC)을 계산합니다.
    
    Args:
        probs (numpy.ndarray): 각 클래스에 대해 예측한 확률입니다.
        labels (numpy.ndarray): 각 예시에 대한 정답 라벨입니다.
    
    Returns:
        float: 모든 클래스의 평균 AUPRC를 백분율로 표시합니다.
    """
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

# F1 score, AUPRC를 위의 함수들을 통해 계산한다.
def compute_metrics(pred):
    """
    예측을 위해 micro F1 score, AUPRC 및 정확도를 계산합니다.

    Args:
        pred: label_ids 및 prediction 값을 포함하는 object입니다.
  
    Returns:
        dict: 'micro F1 scre', 'auprc', '정확도'를 포함한 dict입니다.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
  
    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }