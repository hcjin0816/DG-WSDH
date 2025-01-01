import os
import csv
import json
import numpy as np
import xml.etree.ElementTree as ET
# import openslide
import cv2
from multiprocessing import Pool
import slideio
import glob
def camelyon16xml2json(inxml):
        """
        Convert an annotation of camelyon16 xml format into a json format.
        Arguments:
            inxml: string, path to the input camelyon16 xml format
            outjson: string, path to the output json format
        """
        name = inxml.split('/')[-1][:-4]
        outjson = f'/LiKang/jhc/data/17/camelyon17_annotion_json/{name}.json'
        root = ET.parse(inxml).getroot()
        # annotations_tumor = \
        #     root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        # annotations_0 = \
        #     root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        # annotations_1 = \
        #     root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        # annotations_2 = \
        #     root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
        annotations_tumor = \
            root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="metastases"]')
        annotations_1 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="None"]')
        annotations_positive = \
            annotations_tumor + annotations_0 + annotations_1
        annotations_negative = annotations_2

        json_dict = {}
        json_dict['positive'] = []
        json_dict['negative'] = []

        for annotation in annotations_positive:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for annotation in annotations_negative:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

        tumor_mask(inxml, outjson)
def tumor_mask(xmlpath,json_path):
        level=0
        name = xmlpath.split('/')[-1][:-4]
         # wsi_path
        wsi_path = '' #
        # WSI‘s csv
        save_path = ''
        out_path_csv = ''
        slide = slideio.open_slide(wsi_path) # H W 3
        scene = slide.get_scene(0)  # scene.size:(W,H)
        w, h = scene.size
        mask_tumor = np.zeros((h, w))  # the init mask, and all the value is 0

        # factor = slide.level_downsamples[level]  # get the factor of level * e.g. level 6 is 2^6
        factor = 2**6
        with open(json_path) as f:
            dicts = json.load(f)
        tumor_polygons = dicts['positive']
        for tumor_polygon in tumor_polygons:
            # plot a polygon
            vertices = np.array(tumor_polygon["vertices"])
            vertices = vertices.astype(np.int32)

            cv2.fillPoly(mask_tumor, [vertices], (255))
        scaled_mask = cv2.resize(mask_tumor, None, fx=1 / 64, fy=1 / 64, interpolation=cv2.INTER_NEAREST)
        mask_tumor = mask_tumor[:] > 127

        rect_area = 256 * 256
        datas = {}
        with open(save_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data = []
                bag_id = int(row.get('bag_id'))
                for i in range(8):
                    patch_label = 0
                    y = int(row.get('y'))
                    x = int(row.get('x'))
                    y += i * 256
                    for j in range(8):
                        x += j * 256
                        roi = mask_tumor[y:y + 256, x:x + 256]  # H W
                        intersection_area = np.count_nonzero(roi)
                        intersection_ratio = intersection_area / rect_area
                        if intersection_ratio >= 0.5:
                            print(intersection_area, intersection_ratio, x, y)
                            patch_label = 1
                        data.append(patch_label)

                datas[bag_id]=data
        new_csv_patch_label(save_path, out_path_csv, datas)

def new_csv_patch_label(base_path, out_path_csv,datas):
    with open(base_path, mode='r') as infile, open(out_path_csv, mode='w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # 读取CSV文件的头部
        header = next(reader)
        # 添加要插入的列标题
        header.append('Patch Label')

        # # 写入CSV文件的新头部
        writer.writerow(header)

        # 在这里添加要插入的数据，例如，假设你要添加一列全是示例值的数据
        for row in reader:
            if int(row[0]) in datas.keys():
                row.append(datas[int(row[0])])
                writer.writerow(row)

def csv_patch_label(path):
    datas = [0 for i in range(64)]
    name = path[:-4]
    # 自己的csv地址
    csv_path =''
    # csv输出地址
    out_path_csv = ''
    with open(csv_path, mode='r') as infile, open(out_path_csv, mode='w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # 读取CSV文件的头部
        header = next(reader)
        # 添加要插入的列标题
        header.append('Patch Label')

        # # 写入CSV文件的新头部
        writer.writerow(header)

        # 在这里添加要插入的数据，例如，假设你要添加一列全是示例值的数据
        for row in reader:
            row.append(datas)
            writer.writerow(row)

if __name__=='__main__':
        xml_path = '' # xml文件路径
        json_path = ''  # json文件路径
        os.makedirs(json_path, exist_ok=True)
        li_abspath = glob.glob(f'{xml_path}/*.xml')  # todo
        pool = Pool(1)
        pool.map(camelyon16xml2json, li_abspath)
        pool.close()
        pool.join()

