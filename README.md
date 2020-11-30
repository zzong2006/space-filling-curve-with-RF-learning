# Space Filling Curve (SFC) with Reinforcement Learning (RL)

This is the deep learning project for building the optimal space filling curve with reinforcement learning.

### 폴더 설명
* `reinforcement_learning_review` 
    * 강화 학습 공부를 위해 참고한 파일들을 저장한 폴더 
    * 주로 강화 학습 책에 적힌 코드를 이용했는데, 책 이름이 기억나지 않음
    
* `ToyProblems`
    * `reinforcement_learning_review` 으로 공부한 내용을 토대로 Toy Problem 에 적용한 결과물을 저장한 폴더
    * Toy Problems
        1. `1 ~ n` 사이의 숫자를 중복없이 임의의 순서대로 선택하는 모델 구성 (중복 선택 시 episode 종료): `LSTM_PolicyGradient.py`
        2. 버튼 두 개를 동시에 눌러서 최대의 이득을 얻는 모델 구성: `add_two_value_(algorithm).py` 
        
        
* `library_review`
    * deep learning Framework를 익히기 위해 연습용으로 사용한 파일들을 저장한 폴더 
* `main`   
    * SFC 생성을 위한 강화 학습 모델을 저장한 폴더
    * pyTorch Framework 사용
    * `agent` : 학습 에이전트 모델, 모델의 뼈대가 된다.
    * `rl_network`: 모델의 신경망
    * `environment` : 커브 환경 모듈, `gym` library의 환경과 최대한 비슷하게 구성하려고 했음
    * `driver_{algorithms}` : `environment`에 학습 알고리즘을 각기 다르게 적용한 테스트 드라이브 결과들
        * 성능이 좋은 쪽은 `LSTM + Actor Critic (One-step)` 또는 `REINFORCE (Monte Carlo)`.
    
* `deprecated`
    * 이런 저런 모델을 실험했는데, 성능도 좋지 않고 OOP 느낌이 나지 않아서 버려진 코드들
    * CNN, LSTM, Meta-Learning 등.. 이런 저런 실험을 진행했음


* 서버실 GPU 머신을 pycharm remote interpreter으로 사용 시, 이용 방법 
    * 아래와 같이 port forwading 으로 뚫어주고, pycharm 이용
    ```bash
    ssh -L 6000:<server C IP>:22 <user_serverB>@<server B IP>
    ```
    * server C : 서버실 GPU 머신 (target)
    * server B : gate machine (CUDA)
    * [참고 및 출처](https://stackoverflow.com/questions/37827685/pycharm-configuring-multi-hop-remote-interpreters-via-ssh)
    