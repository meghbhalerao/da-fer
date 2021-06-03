 elif args.source=='AFED': # AFED Train Set
                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6}
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)  
                    data_landmarks.append(landmark)

            elif args.source=='MMI': # MMI Dataset
                
                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            if args.multiple_data=='True':

                if args.target!='CK+': # CK+ Dataset

                    for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                        Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression))
                        for imgFile in Dirs:
                            imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression,imgFile)
                            img = Image.open(imgPath).convert('RGB')
                            ori_img_w, ori_img_h = img.size
                            
                            if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                                continue
                            landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                            
                            data_imgs.append(imgPath)
                            data_labels.append(index)
                            data_bboxs.append((0,0,ori_img_w,ori_img_h))
                            data_landmarks.append(landmark)

                    for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                        Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                        for imgFile in Dirs:
                            imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                            img = Image.open(imgPath).convert('RGB')
                            ori_img_w, ori_img_h = img.size
                            
                            if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                                continue
                            landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                            
                            data_imgs.append(imgPath)
                            data_labels.append(index)
                            data_bboxs.append((0,0,ori_img_w,ori_img_h))
                            data_landmarks.append(landmark)

                if args.target!='JAFFE': # JAFFE Dataset

                    list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1])
                        data_bboxs.append(bbox) 
                        data_landmarks.append(landmark)

                if args.target!='MMI': # MMI Dataset

                    MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                    list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                        data_labels.append(MMItoLabel[list_patition_label[index,1]])
                        data_bboxs.append(bbox) 
                        data_landmarks.append(landmark)

                if args.target!='Oulu-CASIA': # Oulu-CASIA Dataset

                    list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                            continue
                        
                        img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                        ori_img_w, ori_img_h = img.size

                        landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1])
                        data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                        data_landmarks.append(landmark)





if args.target=='CK+': # CK+ Train Set
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression,imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0,0,ori_img_w,ori_img_h))
                        data_landmarks.append(landmark)

            elif args.target=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.target=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.target=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.target=='SFEW': # SFEW Train Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW//Train/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.target=='FER2013': # FER2013 Train Set
                
                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.target=='ExpW': # ExpW Train Set
                
                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/train_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
            elif args.target=='AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)  
                    data_landmarks.append(landmark)

            elif args.target=='WFED': # WFED Train Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/train_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.target=='RAF': # RAF Train Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:5] == "train":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1]-1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)





if args.source=='CK+': # CK+ Val Set
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0,0,ori_img_w,ori_img_h))
                        data_landmarks.append(landmark)

            elif args.source=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.source=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.source=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.source=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.source=='FER2013': # FER2013 Val Set

                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.source=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.source=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.source=='WFED': # WFED Val Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/val_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)


if args.target=='CK+': # CK+ Val Set

                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0,0,ori_img_w,ori_img_h))
                        data_landmarks.append(landmark)

            elif args.target=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.target=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.target=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.target=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.target=='FER2013': # FER2013 Val Set

                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.target=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.target=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.target=='WFED': # WFED Val Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/val_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.target=='RAF': # RAF Test Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:4] == "test":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1]-1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)
