import os

def read_dataset(data_dir):
    if not os.path.exists(data_dir):
        print('Image directory "' + data_dir + '" not found.')
        return None
    directories = ['train', 'test', 'val']
    image_list = {}
    for directory in directories:
        file_list = [file for file in os.listdir(data_dir + '/' + directory) if file.endswith('.png')]
        image_list[directory] = []
        for f in file_list:
            record = {'image': data_dir + '/' + directory + '/' + f,
                      'annotation': data_dir + '/' + directory + 'annot/' + f}
            image_list[directory].append(record)
    
    train_records = image_list['train']
    val_records = image_list['val']
    test_records = image_list['test']

    return train_records, val_records, test_records