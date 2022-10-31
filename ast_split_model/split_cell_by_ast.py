import os
import ast
import sys
import astunparse

# Method to take-out subtrees which are part of "body" node
class MyVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        clusters = []
        for stmt in node.body:
          clusters.append(astunparse.unparse(stmt))
        return clusters

# Method to add "dummy()" function in place of empty-line(as these are not preserved by AST parser)
def add_dummy_code(code_lines):
  for i in range(len(code_lines)):
    if(len(code_lines[i].strip()) == 0):
      j = i+1
      while(j < len(code_lines)):
        if(len(code_lines[j].strip()) != 0):
          gap_length = len(code_lines[j]) - len(code_lines[j].lstrip())
          code_lines[i] = (" " * gap_length) + "dummy()"
          break
        j += 1
      else:
        code_lines[i] += "dummy()"
  return code_lines

# Method to remove unnecessary "\n" created by AST parser
def remove_new_lines(code_lines):
  for i in range(len(code_lines)):
    lines = code_lines[i].split("\n")
    proper_lines = []
    for line in lines:
      if len(line.strip()) != 0:
        proper_lines.append(line)
    code_lines[i] = proper_lines
  return code_lines

# Method to merge clusters which doesn't have empty-line("dummy()") in between
def merge_clusters_by_dummy(clusters):
  clusters_after_merging = [[]]
  list_of_structures = tuple(["def", "for", "while", "if", "else", "class"])
  for cluster in clusters:
    if cluster[0].strip() == "dummy()":
      clusters_after_merging.append([])
    else:
      
      # Don't merge if any of the 2 clusters are 2 structures
      if (len(clusters_after_merging[-1]) != 0 and clusters_after_merging[-1][0].strip().startswith(list_of_structures)) or \
          cluster[0].strip().startswith(list_of_structures):
        clusters_after_merging.append(cluster)
      else:
        clusters_after_merging[-1].extend(cluster)
  
  # Remove empty clusters that might be created above
  after_cleaning = []
  for cluster in clusters_after_merging:
    if len(cluster) != 0:
      after_cleaning.append(cluster)

  return after_cleaning

# Method to split larger clusters by empty-line"dummy()"
def split_clusters_by_dummy(clusters, minimum_length_of_structures):
  new_clusters = []
  for cluster in clusters:
      
    # Find if there is "dummy()" and their count
    has_dummy, count_dummy = False, 0
    for line in cluster:
      if line.strip() == "dummy()":
        has_dummy = True
        count_dummy += 1
    
    if has_dummy :
        
      # Cluster should have more than "minimum_length_of_structures" code lines to break
      if len(cluster) - count_dummy > minimum_length_of_structures :
        splitted_clusters = [[]]
        for line in cluster:
          if line.strip() == "dummy()":
            splitted_clusters.append([])
          else:
            splitted_clusters[-1].append(line)

        # Remove empty clusters that might be created above
        after_cleaning = []
        for cluster in splitted_clusters:
          if len(cluster) != 0:
            after_cleaning.append(cluster)

        new_clusters.extend(after_cleaning)
      else:
        # Just remove the "dummy()" calls
        after_cleaning = []
        for line in cluster:
          if line.strip() == "dummy()":
            pass
          else:
            after_cleaning.append(line)
        new_clusters.append(after_cleaning)
    else:
      new_clusters.append(cluster)
  return new_clusters

def split_code_cell(code_lines, minimum_length_of_structures):
    
    # Add "dummy()" function in place of empty-line
    instrumented_code = add_dummy_code(code_lines)
    print("\nInstrumented: ", instrumented_code)
    
    # Traverse and take-out elements from "body"(to get larger structures)
    # If AST-parser fails once, try again removing "dummy()" if it's there at the error line
    visitor = MyVisitor()
    try:
      clusters = visitor.visit(ast.parse("\n".join(instrumented_code)))
    except Exception as e:
      error_line_number = int(str(e)[str(e).find("line")+5 : -1]) - 1
      if "dummy()" in instrumented_code[error_line_number]:
        instrumented_code[error_line_number] = instrumented_code[error_line_number][: instrumented_code[error_line_number].find("dummy()")]
        clusters = visitor.visit(ast.parse("\n".join(instrumented_code)))
      else:
        raise e
      
    print("\nAfter parsing: ", clusters)
    
    # Remove unnecessary "\n" created by AST parser
    clusters_after_removing_new_lines = remove_new_lines(clusters)
    print("\nAfter removing new-lines: ", clusters_after_removing_new_lines)
    
    # Merge the clusters which doesn't have empty-line("dummy()") in between
    clusters_after_merge_by_dummy = merge_clusters_by_dummy(clusters_after_removing_new_lines)
    print("\nAfter merging by new-line: ", clusters_after_merge_by_dummy)
    print("\nNo of clusters: ", len(clusters_after_merge_by_dummy))
    
    # Split larger clusters by empty-line("dummy()")
    clusters_after_split_by_dummy = split_clusters_by_dummy(clusters_after_merge_by_dummy, minimum_length_of_structures)
    print("\nAfter splitting by dummy(): ", clusters_after_split_by_dummy)
    print("\nNo of clusters: ", len(clusters_after_split_by_dummy))
    
    return clusters_after_split_by_dummy
          
if __name__ == '__main__':
    
    code_lines_1 = [
                "from keras.metrics import top_k_categorical_accuracy",
                "",
                "",
                "def top_5_accuracy(y_true, y_pred):",
                "    return top_k_categorical_accuracy(y_true, y_pred, k=5)",
                "",
                "model = Sequential()",
                "model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (100, 100, 3)))",
                "model.add(Dropout(0.5))",
                "model.add(GlobalMaxPooling2D()) ",
                "model.add(Dense(5005, activation = 'softmax'))",
                "",
                "model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy', top_5_accuracy])",
                "",
                "checkpointer = ModelCheckpoint(filepath='weights.hdf5', ",
                "                               verbose=1, save_best_only=True)",
                "",
                "early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)",
                "",
                "model.fit_generator(generator=train_generator,",
                "                    steps_per_epoch=STEP_SIZE_TRAIN,",
                "                    validation_data=validation_generator,",
                "                    validation_steps=STEP_SIZE_VALID,",
                "                    epochs=2, callbacks = [checkpointer, early_stopping])"
            ]
    code_lines_2 = [
                "C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype=\"float32\")",
                "",
                "def score(y_true, y_pred):",
                "    tf.dtypes.cast(y_true, tf.float32)",
                "    tf.dtypes.cast(y_pred, tf.float32)",
                "    sigma = y_pred[:, 2] - y_pred[:, 0]",
                "    fvc_pred = y_pred[:, 1]",
                "    ",
                "    sigma_clip = tf.maximum(sigma, C1)",
                "    delta = tf.abs(y_true[:, 0] - fvc_pred)",
                "    delta = tf.minimum(delta, C2)",
                "    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )",
                "    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)",
                "    return K.mean(metric)",
                "",
                "def qloss(y_true, y_pred):",
                "    qs = [0.2, 0.50, 0.8]",
                "    q = tf.constant(np.array([qs]), dtype=tf.float32)",
                "    e = y_true - y_pred",
                "    v = tf.maximum(q*e, (q-1)*e)",
                "    return K.mean(v)",
                "",
                "",
                "def mloss(_lambda):",
                "    def loss(y_true, y_pred):",
                "        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)",
                "    return loss"
            ]
    code_lines_3 = [
                "img_path = '../input/train/'",
                "",
                "images = [(whale_img, whale_label) for (whale_img, whale_label) in zip(train_df.Image[:5], train_df.Id[:5])]",
                "",
                "fig, m_axs = plt.subplots(1, len(images), figsize = (20, 10))",
                "for ii, c_ax in enumerate(m_axs):",
                "    c_ax.imshow(imread(os.path.join(img_path,images[ii][0])))",
                "    c_ax.set_title(images[ii][1])"
            ]
    code_lines_4 = [
                "TRAINING_SIZE = 300000",
                "TEST_SIZE = 50000",
                "",
                "train = pd.read_csv(",
                "    '../input/train.csv', ",
                "    skiprows=range(1,184903891-TRAINING_SIZE-TEST_SIZE), ",
                "    nrows=TRAINING_SIZE,",
                "    parse_dates=['click_time']",
                ")",
                "",
                "val = pd.read_csv(",
                "    '../input/train.csv', ",
                "    skiprows=range(1,184903891-TEST_SIZE), ",
                "    nrows=TEST_SIZE,",
                "    parse_dates=['click_time']",
                ")",
                "",
                "y_train = train['is_attributed']",
                "y_val = val['is_attributed']"
    ]

    code_lines_5 = [
                "information = []",
                "",
                "for i in np.arange(0,20,1):",
                "    particle_information = truth[truth.particle_id == particles.iloc[i,0]]",
                "    information.append(particle_information)",
                "",
                "fig = plt.figure(figsize=(25, 8))",
                "gs = gridspec.GridSpec(nrows=2, ncols=3, left=0.05, right=0.48, wspace=0.5, hspace = 0.3)",
                "ax = fig.add_subplot(gs[0:2,0:2], projection = '3d')",
                "ax2 =fig.add_subplot(gs[0,2])",
                "ax3 =fig.add_subplot(gs[1,2])",
                "",
                "for trajectory in information:",
                "    ",
                "    ax.plot(",
                "    xs=trajectory.tx,",
                "    ys=trajectory.ty,",
                "    zs=trajectory.tz, marker='o')",
                "    ",
                "    ax2.scatter( ",
                "    x = trajectory.tx,",
                "    y = trajectory.ty)",
                "    ",
                "    ",
                "    ax3.scatter(",
                "    x= trajectory.tx,",
                "    y = trajectory.tz)",
                "    ",
                "ax.set_xlabel('x (mm)')",
                "ax.set_ylabel('y (mm)')",
                "ax.set_zlabel('z (mm)')",
                "ax.set_title('20 different trajectories')",
                "",
                "ax2.set_xlabel('x (mm)')",
                "ax2.set_ylabel('y (mm)')",
                "ax2.set_title('Detector x-y cross section')",
                "",
                "ax3.set_xlabel('x (mm)')",
                "ax3.set_ylabel('z (mm)')",
                "ax3.set_title('Detector x-z cross section')",
                "plt.show()"
            ]
    
    code_lines_6 = [
                "",
                "binary = []",
                "ordinal = []",
                "numeric = []",
                "",
                "for col in v_cols:",
                "    if train_df[col].value_counts().shape[0] == 2:",
                "        binary.append(col)",
                "    elif train_df[col].sum() - train_df[col].sum().astype('int') == 0:",
                "        ordinal.append(col)",
                "    else:",
                "        numeric.append(col)",
                "        ",
                "print(f'Binary features {len(binary)}: {binary}\\n')",
                "print(f'Ordinal features {len(ordinal)}: {ordinal}\\n')",
                "print(f'Numeric features {len(numeric)}: {numeric}\\n')        "
            ]
    
    code_lines_7 = [
                "def plot_cat(col, rot = 0, n = False, fillna = np.nan, annot = False, show_rate = True):",
                "    ",
                "    ",
                "    ",
                "    rate = (train_df[train_df['isFraud'] == 1][col].fillna(fillna).value_counts() /",
                "            train_df[col].fillna(fillna).value_counts()).sort_values(ascending = False)",
                "    ",
                "    if n:",
                "        order = rate.iloc[:n].index",
                "    else:",
                "        order = rate.index    ",
                "    ",
                "    g1 = sns.countplot(train_df[col].fillna(fillna), hue = train_df['isFraud'], order = order)",
                "    g1.set_ylabel('')",
                "    ",
                "    if annot:",
                "        for p in g1.patches:",
                "            g1.annotate('{:.2f}%'.format((p.get_height() / train_df.shape[0]) * 100, 2), ",
                "                       (p.get_x() + 0.05, p.get_height()+5000))",
                "            ",
                "    plt.xticks(rotation = rot)",
                "    ",
                "    if show_rate:",
                "        g2 = g1.twinx()",
                "        g2 = sns.pointplot(x = rate.index.values, y = rate.fillna(0).values, order = order, color = 'black')",
                "        plt.xticks(rotation = rot)"
            ]
    code_lines_8 = [
                "from sklearn.model_selection import train_test_split",
                "from keras.preprocessing.image import ImageDataGenerator",
                "import tensorflow as tf",
                "import random",
                "import threading",
                "trainImgPath = \"/kaggle/input/severstal-steel-defect-detection/train_images/\"",
                "trainCsv = \"/kaggle/input/severstal-steel-defect-detection/train.csv\"",
                "df1 = pd.read_csv(trainCsv)",
                "df2 = df1[~df1['EncodedPixels'].isnull()].head(7000)",
                "df3 = df1[df1['EncodedPixels'].isnull()].head(200)",
                "df1 = pd.concat([df2,df3])",
                "df1['ImageId'] = df1['ImageId_ClassId'].apply(lambda s:s.split(\"_\")[0])",
                "df1['Labels'] =  df1['ImageId_ClassId'].apply(lambda s:int(s.split(\"_\")[1]))",
                "df1.sample(frac=1)",
                "",
                "getmask  = lambda x: getMaskByClass(x.EncodedPixels, x.Labels)",
                "getimage = lambda img: cv2.resize(cv2.imread(trainImgPath+img),(800,128))",
                "",
                "timin()",
                "class ThreadSafeDataGenerator:",
                "    def __init__(self, it):",
                "        self.it = it",
                "        self.lock = threading.Lock()",
                "",
                "    def __iter__(self):",
                "        return self",
                "",
                "    def __next__(self):",
                "        with self.lock:",
                "            return self.it.__next__()",
                "",
                "def safeItrWrap(f):",
                "    def g(*a, **kw):",
                "        return ThreadSafeDataGenerator(f(*a, **kw))",
                "    return g",
                "",
                "def getDataSlice(labelPassed,  batch_size1, validation_data):",
                "    df = df1.copy()",
                "    if labelPassed is not None:",
                "        df = df[df['Labels']==labelPassed]",
                "    if validation_data:",
                "        randIndex = int(random.randint(df.shape[0]//1.7,df.shape[0] - 70))",
                "        batch_size1=batch_size1*2",
                "    else:",
                "        randIndex = random.randint(0,df.shape[0]//1.5) ",
                "    dfSlice = df.iloc[randIndex:randIndex+batch_size1].copy()",
                "    dfSlice.drop(columns=\"ImageId_ClassId\", inplace=True)",
                "    return dfSlice",
                "",
                "def getMaskByClass(listEncodedString, listLabels):",
                "    mask = np.zeros((256, 1600, 4), dtype=np.int8)",
                "    for encodedString,labels in zip (listEncodedString, listLabels):",
                "        if len(str(encodedString))==0:",
                "            mask[:,:,labels-1] =  np.zeros((256, 1600), dtype=np.int16)",
                "        else:",
                "            encodedString = str(encodedString).split(\" \")",
                "            flatmask = np.zeros(1600*256, dtype=np.int8)",
                "            for i in range(0,len(encodedString)//2):",
                "                start = int(encodedString[2*i])",
                "                end = int(encodedString[2*i]) +int(encodedString[2*i+1])",
                "                flatmask[start:end-1] =  1",
                "            mask[:,:,labels-1] = np.transpose(flatmask.reshape(1600,256))",
                "    return mask",
                "",
                "@safeItrWrap",
                "def getRandomBatch(labelPassed=None, batch_size1=24, validation_data=False):",
                "    while True:",
                "        dfSlice = getDataSlice(labelPassed,  batch_size1, validation_data)",
                "        dfAgg = dfSlice.groupby(['ImageId']).agg({'Labels':list, 'EncodedPixels':list}).reset_index()",
                "        dfAgg[\"EncodedPixels\"] = dfAgg.apply(getmask, axis=1)",
                "        dfAgg = dfAgg.head(batch_size1)",
                "        labels = np.array(dfAgg[\"EncodedPixels\"].tolist()).reshape(dfAgg.shape[0],256,1600,4)",
                "        data =  dfAgg.ImageId.apply(getimage)",
                "        data = np.array(data.tolist(), dtype=np.int16)",
                "        if labelPassed is not None:",
                "            yield data, labels[:,:,:,labelPassed-1].reshape(dfAgg.shape[0],256,1600,1)",
                "        else:",
                "            yield data, labels",
                "",
                "@safeItrWrap",
                "def getRandomTestBatch( batch_size1=24):",
                "    testImgPath = \"/kaggle/input/severstal-steel-defect-detection/test_images/\"",
                "    k = os.listdir(testImgPath)",
                "    while True:",
                "        index = random.randint(0,len(k))",
                "        data =[]",
                "        for iimgh in k[index:index+batch_size1]:",
                "            p = cv2.resize(cv2.imread(testImgPath+iimgh),(800,128))",
                "            data.append(p)",
                "        yield np.array(data).reshape(batch_size1, 800,128,3)",
                "          ",
                "            ",
                "            ",
                "x, y =next(getRandomBatch(1,validation_data=False))   ",
                "timin()",
                "plt.figure(figsize=(35,10))",
                "plt.imshow(x[0], cmap = 'Greys')",
                "plt.figure(figsize=(35,10))",
                "plt.imshow(y[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')",
                "timin()",
                "aug  =  ImageDataGenerator(",
                "",
                "                                             brightness_range=(0.8,1.2), ",
                "                                             fill_mode='nearest', ",
                "                                             horizontal_flip=True, ",
                "                                             rescale=1. / 255, ",
                "                                             dtype=np.int16)"
            ]
    code_lines_9 = [
                "class TGSSaltDataset(data.Dataset):",
                "    ",
                "    def __init__(self, root_path, file_list):",
                "        self.root_path = root_path",
                "        self.file_list = file_list",
                "    ",
                "    def __len__(self):",
                "        return len(self.file_list)",
                "    ",
                "    def __getitem__(self, index):",
                "        if index not in range(0, len(self.file_list)):",
                "            return self.__getitem__(np.random.randint(0, self.__len__()))",
                "        ",
                "        file_id = self.file_list[index]",
                "        ",
                "        image_folder = os.path.join(self.root_path, \"images\")",
                "        image_path = os.path.join(image_folder, file_id + \".png\")",
                "        ",
                "        mask_folder = os.path.join(self.root_path, \"masks\")",
                "        mask_path = os.path.join(mask_folder, file_id + \".png\")",
                "        ",
                "        image = np.array(imageio.imread(image_path), dtype=np.uint8)",
                "        mask = np.array(imageio.imread(mask_path), dtype=np.uint8)",
                "        ",
                "        return image, mask"
            ]
    
    code_lines_10 = [
                "class generator(keras.utils.Sequence):",
                "    ",
                "    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=320, shuffle=True, augment=False, predict=False):",
                "        self.folder = folder",
                "        self.filenames = filenames",
                "        self.pneumonia_locations = pneumonia_locations",
                "        self.batch_size = batch_size",
                "        self.image_size = image_size",
                "        self.shuffle = shuffle",
                "        self.augment = augment",
                "        self.predict = predict",
                "        self.on_epoch_end()",
                "        ",
                "    def __load__(self, filename):",
                "        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array",
                "        msk = np.zeros(img.shape)",
                "        filename = filename.split('.')[0]",
                "        if filename in pneumonia_locations:",
                "            for location in pneumonia_locations[filename]:",
                "                x, y, w, h = location",
                "                msk[y:y+h, x:x+w] = 1",
                "        if self.augment and random.random() > 0.5:",
                "            img = np.fliplr(img)",
                "            msk = np.fliplr(msk)",
                "        img = resize(img, (self.image_size, self.image_size), mode='reflect')",
                "        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5",
                "        img = np.expand_dims(img, -1)",
                "        msk = np.expand_dims(msk, -1)",
                "        return img, msk",
                "    ",
                "    def __loadpredict__(self, filename):",
                "        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array",
                "        img = resize(img, (self.image_size, self.image_size), mode='reflect')",
                "        img = np.expand_dims(img, -1)",
                "        return img",
                "        ",
                "    def __getitem__(self, index):",
                "        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]",
                "        if self.predict:",
                "            imgs = [self.__loadpredict__(filename) for filename in filenames]",
                "            imgs = np.array(imgs)",
                "            return imgs, filenames",
                "        else:",
                "            items = [self.__load__(filename) for filename in filenames]",
                "            imgs, msks = zip(*items)",
                "            imgs = np.array(imgs)",
                "            msks = np.array(msks)",
                "            return imgs, msks",
                "        ",
                "    def on_epoch_end(self):",
                "        if self.shuffle:",
                "            random.shuffle(self.filenames)",
                "        ",
                "    def __len__(self):",
                "        if self.predict:",
                "            return int(np.ceil(len(self.filenames) / self.batch_size))",
                "        else:",
                "            return int(len(self.filenames) / self.batch_size)"
            ]
    code_lines_11 = [
                "def basic_readImg(directory, filename):",
                "    sample = scipyImg.imread(directory + filename, mode='RGB')",
                "    if sample.shape[2] != 3:",
                "        return 'The input must be an RGB image.'",
                "    return sample",
                "def basic_showImg(img, size=4):",
                "    plt.figure(figsize=(size,size))",
                "    plt.imshow(img)",
                "    plt.show()",
                "def basic_writeImg(directory, filename, img):",
                "    misc.imsave(directory+filename, img)"
            ]
    code_lines_12 = [
                "EPOCHS = 150",
                "NNBATCHSIZE = 16",
                "GROUP_BATCH_SIZE = 4000",
                "SEED = 321",
                "LR = 0.001",
                "SPLITS = 5",
                "outdir = 'wavenet_models'",
                "flip = False",
                "noise = False",
                "if not os.path.exists(outdir):",
                "    os.makedirs(outdir)",
                "def seed_everything(seed):",
                "    random.seed(seed)",
                "    np.random.seed(seed)",
                "    os.environ['PYTHONHASHSEED'] = str(seed)",
                "    tf.random.set_seed(seed)"
            ]
    
    code_lines_13 = [
                "from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures",
                "",
                "def preprocessing(X, degree):",
                "",
                "    poly = PolynomialFeatures(degree)",
                "    scaler = MinMaxScaler()  ",
                "    lin_scaler = StandardScaler()",
                "    poly_df = pd.DataFrame(lin_scaler.fit_transform(poly.fit_transform(scaler.fit_transform(X))))",
                "    poly_df['SK_ID_CURR'] = X.index",
                "    poly_df.set_index('SK_ID_CURR', inplace=True, drop=True)",
                "    return poly_df"
            ]
    code_lines_14 = ["from csv import QUOTE_ALL", 
                     "", 
                     "for text_col in text_cols:", 
                     "    test_train[text_col] = test_train[text_col].str.replace('\"', ' ')", 
                     "   "]
    code_lines_15 = ["def convertMatToDictionary(path):", 
                     "    ", 
                     "    try: ", 
                     "        mat = loadmat(path)", 
                     "        names = mat['dataStruct'].dtype.names", 
                     "        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}", 
                     "        ", 
                     "    except ValueError:     ", 
                     "        print('File ' + path + ' is corrupted. Will skip this file in the analysis.')", 
                     "        ndata = None", 
                     "    ", 
                     "    return ndata"]
    code_lines_16 = ["fig = px.density_contour(trainset,", 
                     "                         x ='Percent',", 
                     "                         y ='FVC',", 
                     "                         marginal_x=\"histogram\",", 
                     "                         marginal_y=\"histogram\",", 
                     "                         color='SmokingStatus',", 
                     "                         ", 
                     ")", 
                     "fig.update_layout(title='Relationship between Percent and FVC',", 
                     "                  width=800,", 
                     "                  height=400)", 
                     "fig.show()"]
    
    clusters = split_code_cell(code_lines_2, 10)
    #print("\nFinal Clusters: ", clusters)