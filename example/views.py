print("22222222222222")

# from django.shortcuts import render
# from django.http import HttpResponse, JsonResponse
# from rest_framework.parsers import JSONParser
# from .models import Article
# from .serializer import ArticleSerializer
# from django.views.decorators.csrf import csrf_exempt 
# import base64
# # from django.core.files.base import ContentFile
# from django.shortcuts import render, redirect
# from .forms import ImageUploadForm
# import dlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# import tensorflow as tf
# import numpy as np
# import os                                                                                                              
# import glob
# from imageio import imread,imsave
# import cv2
# import argparse
# from django.conf import settings
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# from django.db import models
# import requests
# import ftplib



# @csrf_exempt
# def article_list(request):

#     if request.method == 'GET':
#         # articles = Article.objects.all()
#         # serializer = ArticleSerializer(articles, many=True)

#         path1 = request.GET.get('path1')
#         path2 = request.GET.get('path2')
#         # nomakeup
#         # https://api.cocoforet.com/Upload/UserData/UserAIData/chsjsj88@naver.com-2024-3-14-134332_1.PNG
       
#         # makeup
#         # https://api.cocoforet.com/Upload/UserData/MakeupImg/vRX916.png
        


#         print(path1)
#         print("##############################")
#         print(path2)


#         path1 = request.GET['path1']
#         path2 = request.GET['path2']
        
       
#         ImageContent1 = "UserUploadImg/"+path1.replace("https://api.cocoforet.com/Upload/UserData/UserAIData/","")
#         ImageContent2 = "UserUploadImg/makeup/"+path2.replace("https://api.cocoforet.com/Upload/UserData/MakeupImg/","")
#         print(ImageContent1)
#         print(ImageContent2)
#         ####################################################
#         #          사진 네이밍 고도화 필요(timestamp)        #
#         ####################################################
        
    
#         ####################################################
#         #          사진 다운로드 
#         # with open(ImageContent1, 'wb') as handle:
#         #     response = requests.get(path1, stream=True)

#         #     if not response.ok:
#         #         print(response)

#         #     for block in response.iter_content(1024):
#         #         if not block:
#         #             break

#         #         handle.write(block)

#         # with open(ImageContent2, 'wb') as handle:
#         #     response = requests.get(path2, stream=True)

#         #     if not response.ok:
#         #         print(response)

#         #     for block in response.iter_content(1024):
#         #         if not block:
#         #             break

#         #         handle.write(block)

#         # 이미지 다운로드
#         def download_image(url, save_path):
#             with open(ImageContent1, 'wb') as handle:
#                 response = requests.get(path1, stream=True)

#                 if not response.ok:
#                     print(response)

#                 for block in response.iter_content(1024):
#                     if not block:
#                         break

#                     handle.write(block)

#             with open(ImageContent2, 'wb') as handle:
#                 response = requests.get(path2, stream=True)

#                 if not response.ok:
#                     print(response)

#                 for block in response.iter_content(1024):
#                     if not block:
#                         break

#                     handle.write(block)
#             return True

#         ####################################################
                
#         def process_image(image_file):
#             nparr = np.frombuffer(image_file.read(), np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             return image
            

#         response = requests.get(path1, stream=True)
#         print(response.status_code)

#         predictor_path = 'C:/Users/MS/Desktop/coco/coco/ImgUpload/models/shape_predictor_5_face_landmarks.dat'



#         # 이미지 다운로드 및 처리
#         def download_and_process_image(url, save_path):
#             if download_image(url, save_path):
#                 with open(save_path, 'rb') as file:

#                     print("!@!@@!@!@@!!@@@!@!@@@!@!!@@@!@")
                    
#                     image = process_image(file)
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                     print(image)
#                     detector = dlib.get_frontal_face_detector()
#                     print(detector)
                    
#                     print("~!~!~~!~~!~~~!~~!~!~~!")
#                     # 여기서 이미지를 처리하고 원하는 작업을 수행합니다.

                                
#                     print("###########              start               ##########")
#                     sess = tf.Session()
#                     sess.run(tf.global_variables_initializer())
#                     #plt.show()
#                     saver = tf.train.import_meta_graph('C:/Users/MS/Desktop/coco/coco/ImgUpload/models/model.meta')

#                     print("~~~~~~~~~~~~~~~~~~~~~~~~")
#                     saver.restore(sess, tf.train.latest_checkpoint('C:/Users/MS/Desktop/coco/coco/ImgUpload/models'))
#                     print("@@@@@@@~~!~!~!~!!~")
#                     graph = tf.get_default_graph()

#                     X = graph.get_tensor_by_name('X:0') # source
#                     Y = graph.get_tensor_by_name('Y:0') # reference
#                     Xs = graph.get_tensor_by_name('generator/xs:0') # output

#                     def preprocess(img):
#                         return img.astype(np.float32) / 127.5 - 1.

#                     def postprocess(img):
#                         return ((img + 1.) * 127.5).astype(np.uint8)
                    

#                     print("3333333333333333")

#                     img1 = image
                    
#                     print("3.25")                                                         
#                     img1_faces = align_faces(img1)

#                     print("3.5")

#                     img2 = dlib.load_rgb_image('C:/Users/MS/Desktop/coco/coco/ImgUpload/imgs/makeup/vFG756.png')
#                     img2_faces = align_faces(img2)

#                     print('3.6')
#                     fig, axes = plt.subplots(1, 2, figsize=(16, 10))
#                     axes[0].imshow(img1_faces[0])
#                     axes[1].imshow(img2_faces[0])

#                     src_img = img1_faces[0]
#                     ref_img = img2_faces[0]

#                     print("4444444444")

#                     X_img = preprocess(src_img)
#                     X_img = np.expand_dims(X_img, axis=0)


#                     Y_img = preprocess(ref_img)
#                     Y_img = np.expand_dims(Y_img, axis=0)

#                     print(Y_img)

#                     print("7777777777777")

#                     output = sess.run(Xs, feed_dict={
#                         X: X_img,
#                         Y: Y_img
#                     })

#                     output_img = postprocess(output[0])
#                     print(output_img)

#                     fig, axes = plt.subplots(1, 3, figsize=(20, 10))
#                     axes[0].imshow(src_img)
#                     axes[1].imshow(ref_img)
#                     axes[2].imshow(output_img)
                
#                     plt.imshow(output_img)
#                     plt.show()

#                     # print("##@@!!")
#                     # session = ftplib.FTP()
#                     # print("FTP")
#                     # session.connect('13.125.207.70', 21) # 두 번째 인자는 port number
#                     # session.login("chsjsj88", "@@aszx9494")   # FTP 서버에 접속
#                     # print("FTP2")
#                     # uploadfile = open('C:/Users/MS/Desktop/coco/coco/pic1.jpg' ,mode='rb') #업로드할 파일 open
                    
#                     # print(uploadfile)
#                     # session.encoding='utf-8'
#                     # session.storbinary('STOR ' + '/MakeupResult/test.jpg', wf) #파일 업로드
                    
#                     # uploadfile.close() # 파일 닫기
                    
#                     # session.quit() # 서버 나가기
#                     # print('파일전송함')



#                     print("END!")

#                     return image
#             else:
#                 return None
            

#         def align_faces(img):
#             # print("1")
#             detector = dlib.get_frontal_face_detector()
#             # print("2")
#             dets = detector(img, 1)
#             # print("3")
#             objs = dlib.full_object_detections()
#             # print('4')

#             sp = dlib.shape_predictor(predictor_path)
#             # print(sp)
            
#             for detection in dets:
#                 s = sp(img, detection)
#                 objs.append(s)
                
#             # print("5")
#             faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
#             # print("6")
#             return faces
        

#         # download_and_process_image(path1, ImageContent1)

    
#         result = {"RegMsg": "SUCCESS1"}

#         return JsonResponse(result)

#     elif request.method == 'POST':
        
#         Image1 = request.POST.get('Image1')
#         Image2 = request.POST.get('Image2')
#         print(Image1)
#         print(Image2)

#         result = {"RegMsg": "SUCCESS2"}

#         # Return the response as JSON
#         return JsonResponse(result)