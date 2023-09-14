import os

import cv2
from ocr_mods.dataset import RawDataset, AlignCollate
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def read_number(image, model_ocr, device, converter):
    if not os.path.exists('test_ocr'):
        os.makedirs('test_ocr')
    cv2.imwrite(f'test_ocr/dummy.jpg', image)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False)
    demo_data = RawDataset(root='test_ocr')  # use RawDataset

    demo_loader = DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        # num_workers=int(4),
        num_workers=0,
        collate_fn=AlignCollate_demo, pin_memory=True)

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            # print(image_tensors.shape)
            batch_size = image_tensors.size(0)
            # print(batch_size)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([1] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, 1 + 1).fill_(0).to(device)


            preds = model_ocr(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            # with open('demo_image/validation/gt.txt', 'a') as f:
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):

                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                # f.write(img_name+' '+pred+'\n')
                os.system(f'rm -r test_ocr/')
                return (pred, confidence_score)
        os.system(f'rm -r test_ocr/')
        return '', 0
