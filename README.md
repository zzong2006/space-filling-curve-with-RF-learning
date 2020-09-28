# space-filling-curve-with-RF-learning

This is the python project for building space filling curve with reinforcement learning.


* `tensorflow_curve` 폴더
    * TensorFlow library를 활용하여 space filling curve를 구축 시도한 프로그램
* `reinforcement_learning_review` 폴더
    * 강화 학습 공부를 위해 참고한 파일들
   
   
   
* 서버실 GPU 머신을 pycharm remote interpreter으로 사용 시, 이용 방법 
    * 아래와 같이 port forwading 으로 뚫어주고, pycharm 이용
    ```bash
    ssh -L 6000:<server C IP>:22 <user_serverB>@<server B IP>
    ```
    * server C : 서버실 GPU 머신 (target)
    * server B : gate machine (CUDA)
    * [참고 및 출처](https://stackoverflow.com/questions/37827685/pycharm-configuring-multi-hop-remote-interpreters-via-ssh)
    