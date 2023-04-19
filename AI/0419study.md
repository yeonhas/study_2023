# 학습 learning
일반화:주어진 정보를 정리하여 다른 데이터에도 적용할 수 있도록 만드는 기술

입력에 대해 단순 암기가 아님


# 딥러닝 모델
여러 개의 네트워크 층으로 구성

주어진 데이터를 잘 분류할 수 있도록 층과 층 사이의 변환 연산을 개선해 나가는 과정


## 범주형 데이터의 수치화 one-hot-encoding, to categorical

## 수치형 데이터의 정규화 0에서 1까지의 값으로 nomalize

` (value-min)/(max-min) `


```
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
data_train["Sex"] #성별

#categorical을 numerical로 바꿔줌
sex_num = np.zeros( [len(data_train), ]) #len(data_train.shape[0]
sex_num[data_train["Sex"] == 'female']= 1

#one-hot-encoding으로 바꿔줌
from tensorflow.keras.utils import to_categorical
sex_categ = to_categorical(sex_num)

pclass_categ = to_categorical(data_train['Pclass'] - 1)
age_num = data_train['Age']/max(data_train['Age'])

np.isnan(age_num)
age_num[np.isnan(age_num)]=0.5

sibsp_num = data_train['SibSp']/max(data_train['SibSp'])
parch_num = data_train['Parch']/max(data_train['Parch'])

#5가지의 데이터를 합쳐줌 concatenate사용
#dimension을 통일시켜줌 expand_dims사용
np.concatenate((sex_categ, pclass_categ, np.expand_dims(age_num, 1), np.expand_dims(sibsp_num, 1), np.expand_dims(parch_num, 1)), 1)
#data_train_np로 저장한 후 shape 확인

#합쳐줄 numpy array를 미리 만들어 놓고 data를 바로 저장
data_train_np = np.zeros([data_train.shape[0], 8])
data_train_np[:, :2] = to_categorical(sex_num)
data_train_np[:, 2:5] = to_categorical(data_train['Pclass']-1)

data_train_np[:,5] = data_train['Age']/80
data_train_np[:,6] = data_train['SibSp']/10
data_train_np[:,7] = data_train['Parch']/10

#정답데이터 생성
data_train_np_y = to_categorical(data_train['Survived'])

```

# 데이터의 정리

학습을 위한 데이터 data_train_np

알려져 있는 결과 data_train_np_y

결과를 추정하려는 데이터 data_test_np

추정한 결과 data_test_np_y
