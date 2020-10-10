# Space Filling Curve (SFC) with Reinforcement Learning (RL)

This is the deep learning project for building the optimal space filling curve with reinforcement learning.

### 폴더 설명
* `tensorflow_curve`
    * TensorFlow library를 활용하여 space filling curve를 구축 시도한 프로그램들을 담은 폴더
* `reinforcement_learning_review` 
    * 강화 학습 공부를 위해 참고한 파일들을 저장한 폴더
    * ㄹㄷㅈㄷ
    * 
* `library_review`
    * deep learning Framework를 익히기 위해 연습용으로 사용한 파일들을 저장한 폴더 
* `main`   
    * SFC 생성을 위한 강화 학습 모델을 저장한 폴더
    * pyTorch Framework 사용
    * 주로 `main({B}_{A}).py` 형식으로 파일들이 저장됨
        * `{A}`: 사용한 강화 학습 알고리즘들 
            * Actor-Critic, Policy Gradient, Q-Learning, Proximal Policy Optimization(PPO), Meta Learning for RL
        * `{B}`: 사용한 신경망 (`{B}`는 표기가 되어있지 않은 경우가 많음)
            * Convolutional Neural Network (CNN),Deep Neural Network(DNN), Long Short-Term Memory Model(LSTM)
    
* 서버실 GPU 머신을 pycharm remote interpreter으로 사용 시, 이용 방법 
    * 아래와 같이 port forwading 으로 뚫어주고, pycharm 이용
    ```bash
    ssh -L 6000:<server C IP>:22 <user_serverB>@<server B IP>
    ```
    * server C : 서버실 GPU 머신 (target)
    * server B : gate machine (CUDA)
    * [참고 및 출처](https://stackoverflow.com/questions/37827685/pycharm-configuring-multi-hop-remote-interpreters-via-ssh)
    