#Trained and tested on Python 3.8...
after setting up the environment according to requirements.txt
#use command:
pip install -r requirements.txt
#then:
python3 inference.py

#######
Note: The script takes folder as input such as user_img folder is used as input and a json with all the results is saved by name of Bank_Card_Results.json , Also detections of the cards are saved in results folder with corresponding input name. If you want to change input folder in script. Change it in line # 163 of inference.py
