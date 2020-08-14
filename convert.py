import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from PIL import Image
import os
import argparse
import sys

def createAnnotationPascalVocTree(folder, basename, path, width, height):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = basename
    #ET.SubElement(annotation, 'path').text = path

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'

    ET.SubElement(annotation, 'segmented').text = '0'

    return ET.ElementTree(annotation)

def createObjectPascalVocTree(xmin, ymin, xmax, ymax, image_width, image_height):
    obj = ET.Element('object')
    ET.SubElement(obj, 'name').text = 'person'
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'

    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(max(xmin,0))
    ET.SubElement(bndbox, 'ymin').text = str(max(ymin,0))
    ET.SubElement(bndbox, 'xmax').text = str(min(xmax,image_width))
    ET.SubElement(bndbox, 'ymax').text = str(min(ymax,image_height))

    return ET.ElementTree(obj)

def parseImFilename(imFilename, imFolder):
    im = Image.open(os.path.join(imFolder, imFilename))
            
    folder, basename = imFilename.split('/')
    width, height = im.size

    return folder, basename, imFilename, width, height

def convertWFAnnotations(annotationsPath, targetDir, imFolder, validOnly, occlusionOptions, blurOptions):
    ann = None
    basename = ''
    no_obj_img = []
    xml_counter = 0
    with open(annotationsPath) as f:
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        line = f.readline().strip()
        while line:
            imFilename = line
            folder, basename, path, width, height = parseImFilename(imFilename, imFolder)
            ann = createAnnotationPascalVocTree(folder, basename, os.path.join(imFolder, path), width, height)
            nbBndboxes = f.readline().strip()
            
            i = 0
            foundValidBoxes = False
            while i < int(nbBndboxes):
                i = i + 1
                x1, y1, w, h, blur, expr, illum, invalid, occl, pose = [int(k) for k in f.readline().split()]
                x2 = x1 + w
                y2 = y1 + h

                if (x2 <= x1 or y2 <= y1):
                    print('[WARNING] Invalid box dimensions in image "{}" x1 y1 w h: {} {} {} {}'.format(imFilename,x1,y1,w,h))
                    continue

                if validOnly and invalid == 1:
                    continue

                if not occl in occlusionOptions:
                    continue

                if not blur in blurOptions:
                    continue

                foundValidBoxes = True
                ann.getroot().append(createObjectPascalVocTree(x1, y1, x2, y2, width, height).getroot())

            xmlstr = minidom.parseString(ET.tostring(ann.getroot())).childNodes[0].toprettyxml(indent="    ")
            annFilename = os.path.join(targetDir, basename.replace('.jpg','.xml'))

            if ann.find("object") == None:
                no_obj_img.append(line)
            if foundValidBoxes:
                xml_counter += 1
                with open(annFilename,"w") as sf:
                    sf.write(xmlstr)
            print('{} => {}'.format(basename, annFilename))
            line = f.readline().strip()


    for i in no_obj_img:
        print('[WARNING] "{}" contains no object and is not converted.'.format(i))
    print("Created {} xml".format(xml_counter))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert WIDER annotations to VOC format')
    parser.add_argument('-ap', dest='annotations_path', required=True, help='the annotations file path. ie: "-ap ./wider_face_split/wider_face_train_bbx_gt.txt".')
    parser.add_argument('-td', dest='target_dir', required=True, help='the target directory where XML files will be saved. ie: "-td ./WIDER_train_annotations"')
    parser.add_argument('-id', dest='images_dir', required=True, help='the images directory. ie:"-id ./WIDER_train/images"')
    parser.add_argument('-vl', dest="valid", action='store_true', default=False, help='Only include valid boxes from WIDERFACE annotation')
    parser.add_argument('-oc', dest="occlusion", nargs='+', default=[0, 1, 2], choices=[0, 1, 2], type=int, help='Filter boxes by "occlusion" flag: no occlusion->0, partial occlusion->1, heavy occlusion->2; ie: "-oc 0 1"')
    parser.add_argument('-bl', dest="blur", nargs='+', default=[0, 1, 2], choices=[0, 1, 2], type=int, help='Filter boxes by "blur" flag: clear->0, normal blur->1, heavy blur->2; ie: "-bl 0 1"')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    convertWFAnnotations(args.annotations_path, args.target_dir, args.images_dir, args.valid, args.occlusion, args.blur)