use tensorboard:

~~~bash
tensorboard --logdir logs/parkour_new/galileo_teacher/tensorboard/
~~~

play:

~~~
python legged_gym/legged_gym/scripts/play.py --task galileo --exptid galileo_teacher --terrain_mode demo --checkpoint -1
~~~

