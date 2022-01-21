import pickle as pkl
import json


def process_annot(anno_path):
    print("Processing: " + anno_path)
    print("Reading annotation..")
    with open(anno_path) as f:
        data = json.load(f)

    out_path = anno_path.replace('.json', '.pkl')

    images = data['images']
    annotations = data['annotations']

    assert len(images) == len(annotations)

    images_anno = {}
    for img, anno in zip(images, annotations):
        assert img['id'] == anno['id']
        img['anno'] = anno
        images_anno[img['id']] = img

    print("Saving processed data..")
    with open(out_path, 'wb') as f:
        pkl.dump(images_anno, f)

    print('Saved to ' + out_path)


anno_path = r'\\105.1.1.1\Hand\HO&HH\InterHand2.6M\downloads\InterHand2.6M.annotations.5.fps/annotations/train/InterHand2.6M_train_data.json'
process_annot(anno_path)

anno_path = r'\\105.1.1.1\Hand\HO&HH\InterHand2.6M\downloads\InterHand2.6M.annotations.5.fps/annotations/val/InterHand2.6M_val_data.json'
process_annot(anno_path)

anno_path = r'\\105.1.1.1\Hand\HO&HH\InterHand2.6M\downloads\InterHand2.6M.annotations.5.fps/annotations/test/InterHand2.6M_test_data.json'
process_annot(anno_path)
