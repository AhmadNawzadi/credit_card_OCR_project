from PIL import Image
from utils.torch_utils import select_device
from models.experimental import attempt_load
from numpy import random
import time
import dlib
import os
import cv2
import easyocr
import torch
import numpy as np
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from ocr_mods.utils_ocr import AttnLabelConverter
from ocr_mods.model import Model
from ocr_mods.text_utils import read_number
from datetime import datetime
import json
from flask import render_template, url_for, flash, redirect, request
from ocr_package import app, db, bcrypt
from ocr_package.forms import RegistrationForm, LoginForm
from ocr_package.model import User, Card
from flask_login import login_user, current_user, logout_user, login_required

def take_first(tup):
    return tup[0]


def take_second(tup):
    return tup[1]
    
reader = easyocr.Reader(['en'])
Prediction = 'Attn'
Transformation = 'TPS'
FeatureExtraction = 'ResNet'
SequenceModeling = 'BiLSTM'
character = '0123456789/'
weights_ocr = 'weights/ocr_model_94.pth'
device = torch.device('cpu')

converter = AttnLabelConverter(character)
num_class = len(converter.character)
model_ocr = Model(Prediction, Transformation, FeatureExtraction, SequenceModeling, 32, 100, num_class)

model_ocr = torch.nn.DataParallel(model_ocr)

model_ocr.load_state_dict(torch.load(weights_ocr, map_location=device))
model_ocr.eval()
predictor = dlib.shape_predictor('weights/credit_card.dat')

half = device.type != 'cpu'  # half precision only supported on CUDA 
classes=None
agnostic_nms=False
conf_thres=0.25
iou_thres=0.01
augment=False


def shape_to_np(shape):
    coors = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, shape.num_parts):
        coors[i] = (shape.part(i).x, shape.part(i).y)
    return coors


def get_landmarks(img):
    h, w, c = img.shape #c = channels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rect = dlib.rectangle(1, 1, int(w - 2), int(h - 2))
    shape = predictor(gray, rect)

    shape = shape_to_np(shape)
    return shape


def detect(img0, model, imgsz,names):

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    img = letterbox(img0, 640, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    out_data = {'name': (), 'date': [], 'numbers': []}
    box_count = 0
    date_count = 0
    name_flag = False
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                lbl = label.split(' ')[0]
                if lbl == 'name' and name_flag == False:
                    out_data['name'] = (x1, y1, x2, y2)
                    name_flag = True
                elif lbl == 'id' and box_count < 4:
                    out_data['numbers'].append((x1, y1, x2, y2))
                    box_count += 1
                elif lbl == 'date' and date_count < 2:
                    out_data['date'].append((x1, y1, x2, y2))
                    date_count += 1
    return out_data


def warpPerspective(landmarks, crop_upper_):
    src_pts_1 = np.array([
        landmarks[0], landmarks[1], landmarks[2], landmarks[3]],
        dtype=np.float32)

    max_x = max([[abs(landmarks[0][0] - landmarks[2][0])], [abs(landmarks[1][0] - landmarks[3][0])]])[0]
    max_y = max([[abs(landmarks[0][1] - landmarks[2][1])], [abs(landmarks[1][1] - landmarks[3][1])]])[0]

    dst_pts = np.array([[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]], dtype=np.float32)
    perspect_1 = cv2.getPerspectiveTransform(src_pts_1, dst_pts)
    cropped_pic1 = cv2.warpPerspective(crop_upper_, perspect_1, (max_x, max_y))
    return cropped_pic1


def get_name_ocr(img):
    result = reader.readtext(img)
    text_name = ' '.join(r[1] for r in result)
    return text_name


@app.route('/#about', methods=['GET'])
def about():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegistrationForm() 
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(name=form.name.data, surname=form.surname.data, email=form.email.data, phone=form.phone.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        message = flash(f'Compte créé avec succès pour {form.name.data}!', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html', form=form)
    
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('ocr'))
        else:
            flash('E-mail ou mot de passe incorrect!', 'danger')
    return render_template('signin.html', title='Signin', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/ocr', methods=['GET', 'POST'])
@login_required
def ocr():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file') 
        picture = Image.open(file) 
        rgb_im = picture.convert('RGB')
        #filename = os.rename(rgb_im, 'card_1.jpg')
        filename = 'card_1.jpg' 
        picture = rgb_im.save("user_imgs/"+ filename)
        device='cpu'
        imgsz=640
        weights='weights/credit_card_959_latest.pt'
        device = select_device(device)
        #half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model----------------------------------------------
        model = attempt_load(weights, map_location=device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names
        #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        #t1 = time.time()
        final_data = {}

        for root , dir, files in os.walk('user_imgs', topdown=True):
            files = [file for file in files if not file.startswith('.') and os.path.isfile(os.path.join(root, file))]
           
            numbers_list = [int(num.split('_')[1].split('.')[0]) for num in files]
            ul_list = [(files[i], numbers_list[i]) for i in range(len(files))]
            sorted_list = sorted(ul_list, key=take_second)
            final_files = [i[0] for i in sorted_list]
            for file in final_files:
                print('Processing:', file)
                img_name = file.split('.')[0]
                file_path = os.path.join(root, file)
                save_path = os.path.join('ocr_package/static', img_name+'.jpg')
                main_img = cv2.imread(file_path)
                main_img = cv2.resize(main_img, (960, 520))
                crop_upper_ = main_img.copy()
                landmarks = get_landmarks(main_img)
                tl = round(0.002 * (main_img.shape[0] + main_img.shape[1]) / 2) + 1
                Perspective_img = warpPerspective(landmarks,crop_upper_)
                extracted_card = Perspective_img.copy()
                output = detect(Perspective_img, model, imgsz, names)
                name_text = f''
                bank_name = output['name']
                if bank_name == ():
                    name_text = 'Not Detected...'
                else:
                    # print('Reading Bank Name')
                    name_image = Perspective_img[bank_name[1]:bank_name[3], bank_name[0]:bank_name[2]]
                    name_text = get_name_ocr(name_image).upper()
                    cv2.rectangle(extracted_card, (bank_name[0], bank_name[1]), (bank_name[2], bank_name[3]),
                                (0, 255, 0), tl)

                number_boxes = output['numbers']
                full_number = f''
                if len(number_boxes) != 4:
                    print('Card Number not Detected Correctly')
                    for box in number_boxes:
                        ext = Perspective_img[box[1]: box[3], box[0]: box[2]]

                        # cv2.imwrite(f'extracted_numbers/number_img_{num_count}.jpg', ext)
                        # num_count += 1
                        cv2.rectangle(extracted_card, (box[0], box[1]), (box[2], box[3]),
                                    (0, 0, 255), tl)
                else:
                    # print('Reading Card Number')
                    number_boxes = sorted(number_boxes, key=take_first)
                    for box in number_boxes:
                        ext = Perspective_img[box[1]: box[3], box[0]: box[2]]
                        num_output = read_number(ext, model_ocr, device, converter)[0]
                        full_number += f'{num_output} '

                        # cv2.imwrite(f'extracted_numbers/number_img_{num_count}.jpg', ext)
                        # num_count += 1
                        cv2.rectangle(extracted_card, (box[0], box[1]), (box[2], box[3]),
                                    (0, 0, 255), tl)
                if full_number != '':
                    full_number = full_number[:-1]

                full_date = f''
                date_boxes = output['date']
                if len(date_boxes) < 1:
                    print('Date Not Correctly Detected')
                    for box in date_boxes:
                        ext = Perspective_img[box[1]: box[3], box[0]: box[2]]
                        # cv2.imwrite(f'extracted_numbers/number_img_{num_count}.jpg', ext)
                        # num_count += 1
                        cv2.rectangle(extracted_card, (box[0], box[1]), (box[2], box[3]),
                                    (255, 0, 0), tl)
                else:
                    date_boxes = sorted(date_boxes, key=take_first)
                    # print('Reading Date')
                    for box in date_boxes:
                        ext = Perspective_img[box[1]: box[3], box[0]: box[2]]
                        date_number = read_number(ext, model_ocr, device, converter)[0]
                        full_date += f'{date_number} '
                        # cv2.imwrite(f'extracted_numbers/number_img_{num_count}.jpg', ext)
                        # num_count += 1
                        cv2.rectangle(extracted_card, (box[0], box[1]), (box[2], box[3]),
                                    (255, 0, 0), tl)
                if full_date != '':
                    full_date = full_date[:-1]
                    if ' ' in full_date:
                        if '/' in full_date.split(' ')[0] and '/' in full_date.split(' ')[-1]:
                            full_date = full_date.split(' ')[-1]
                        elif '/' in full_date.split(' ')[0] or '/' in full_date.split(' ')[-1]:
                            full_date = full_date.split(' ')[0] if '/' in full_date.split(' ')[0] \
                                else full_date.split(' ')[-1]
                        else:
                            full_date = full_date.split(' ')[0]
                    if '/' in full_date:
                        month_d = int(full_date.split('/')[0])
                        year_d = int(full_date.split('/')[-1])
                        dt = str(datetime.now())
                        year = int(dt.split()[0].split('-')[0][2:])
                        if month_d > 12 or year_d > year + 10:
                            full_date = 'Not Recognized...'
                else:
                    full_date = 'Not Detected...'

                output_texts = {'Card_Holder': name_text, 'Card_Number': full_number, 'Valid_Date': full_date}
                final_data[file.split('.')[0]] = output_texts
                print(output_texts)
                cv2.imwrite(save_path, extracted_card)

        with open('Bank_Card_Results.json', 'w') as json_f:
            json.dump(final_data, json_f, indent=4)

        with open('Bank_Card_Results.json', 'r') as myfile:
            #json_data = myfile.read()
            data = json.load(myfile)
            card_holder = data['card_1']['Card_Holder']
            card_number = data['card_1']['Card_Number']
            valid_date = data['card_1']['Valid_Date']
        return render_template('ocr.html', result_image=save_path, card_holder=card_holder, card_number=card_number, valid_date=valid_date)

    return render_template('ocr.html')

@app.route('/savecard', methods=['GET', 'POST'])
def savecard():
    filename = 'card_1.jpg'
    with open('Bank_Card_Results.json', 'r') as myfile:
        data = json.load(myfile)
        card_holder = data['card_1']['Card_Holder']
        card_number = data['card_1']['Card_Number']
        valid_date = data['card_1']['Valid_Date']
        card = Card(card_number=card_number, card_holder=card_holder, valid_date=valid_date, image_file=filename, user_id=current_user.id)
        db.session.add(card)
        db.session.commit()
    flash(f'Carte a été ajouté avec succès !', 'success')
    return redirect(url_for('ocr'))