import os
import sys

# Merge the clusters by if-elif-else
def split_by_if_elif_else(code_lines):
    pass

# Merge the clusters by if-else, while-else, for-else
def split_by_if_else(code_lines):
    pass

# Create initial cluster involving each line to one cluster
def assign_cluster_to_each_line(code_lines):
    clusters = []
    for line_number in range(len(code_lines)):
        clusters.append([code_lines[line_number]])
    return clusters

# Combine multiple empty lines to one
def combine_multiple_empty_lines_to_one(clusters):
    new_clusters = []
    for cluster_number in range(len(clusters)):
        if(len(clusters[cluster_number][0].strip()) != 0):
            new_clusters.append(clusters[cluster_number])
        else:
            if len(new_clusters) == 0 or len(new_clusters[-1][0].strip()) == 0:
                pass
            else:
                new_clusters.append(clusters[cluster_number])
    return new_clusters

# Merge clusters by definitions, forr-loop, while-loop
def split_by_def_for_while(clusters):
    merge_ranges, cluster_number = [], 0
    
    # Get the range of clusters that need to get merged
    prefix_tuple = ("def", "while", "for")
    while(cluster_number < len(clusters)):
        if(clusters[cluster_number][0].strip().startswith(prefix_tuple)):
            gap_length = -1
            first_code_line_number = cluster_number + 1
            while(first_code_line_number < len(clusters)):
                if(len(clusters[first_code_line_number][0].strip()) != 0):
                    gap_length = len(clusters[first_code_line_number][0]) - len(clusters[first_code_line_number][0].lstrip())
                    break
                first_code_line_number += 1
            last_code_line_number = first_code_line_number + 1
            while(last_code_line_number < len(clusters)):
                if(len(clusters[last_code_line_number][0].strip()) != 0):
                    current_gap_length = len(clusters[last_code_line_number][0]) - len(clusters[last_code_line_number][0].lstrip())
                    if(current_gap_length < gap_length):
                        break
                last_code_line_number += 1
            if(len(clusters[last_code_line_number - 1][0].strip()) == 0):
                merge_ranges.append([cluster_number, last_code_line_number - 2])
            else:
                merge_ranges.append([cluster_number, last_code_line_number - 1])
            cluster_number = last_code_line_number
        else:
            cluster_number += 1
    
    # Merge the clusters as per the merge_ranges
    new_clusters = []
    current_cluster_number = 0
    for merge in merge_ranges:
        
        # Add the clusters before merge
        if(current_cluster_number < merge[0]):
            for cluster_number in range(current_cluster_number, merge[0]):
                new_clusters.append(clusters[cluster_number])
            current_cluster_number = merge[0]
        
        # Create new cluster and add
        new_cluster = []
        for cluster_number in range(merge[0], merge[1] + 1):
            new_cluster += clusters[cluster_number]
        new_clusters.append(new_cluster)
        current_cluster_number = merge[1] + 1
        
    # Add the last remaining clusters
    if(current_cluster_number < len(clusters)):
        for cluster_number in range(current_cluster_number, len(clusters)):
                new_clusters.append(clusters[cluster_number])
        current_cluster_number = len(clusters)
            
    return new_clusters

# Merge the clusters by new-line/empty-line 
def split_by_comment_or_newline(clusters):
    new_clusters = []
    for cluster_number in range(len(clusters)):
        if(len(clusters[cluster_number][0].strip()) != 0):
            if len(new_clusters) == 0:
                new_clusters.append(clusters[cluster_number])
            else:
                new_clusters[-1] += clusters[cluster_number]
        else:
            new_clusters.append([])
    
    # Remove the possible empty list in the end
    if len(new_clusters[-1]) == 0:
        return new_clusters[:-1]
    else:
        return new_clusters

def split_code_cell(code_lines):
    clusters = assign_cluster_to_each_line(code_lines)
    clusters = combine_multiple_empty_lines_to_one(clusters)
    clusters = split_by_def_for_while(clusters)
    clusters = split_by_comment_or_newline(clusters)
    print(clusters)
    
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
    split_code_cell(code_lines_3)