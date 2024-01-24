import pandas as pd

def undersampling():
    # './EDA/train_split_v1.csv' 파일을 읽어와서 DataFrame으로 변환
    file_path = '../dataset/train/train_split_v1.csv'
    df = pd.read_csv(file_path)

    # 'no_relation' 레이블을 가진 데이터 중 25%를 무작위로 선택하여 삭제하여 언더샘플링 수행
    percentage_to_delete = 0.35
    no_relation_rows = df[df['label'] == 'no_relation']
    num_rows_to_delete = int(len(no_relation_rows) * percentage_to_delete)
    rows_to_delete = no_relation_rows.sample(n=num_rows_to_delete, random_state=42)
    df = df.drop(rows_to_delete.index)

    # 언더샘플링이 적용된 DataFrame을 './EDA/train_split_v2.csv' 파일로 저장
    new_file_path = '../dataset/train/train_split_v2.csv'
    df.to_csv(new_file_path, index=False)
    
if __name__ == '__main__':
    undersampling()