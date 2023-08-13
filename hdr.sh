mkdir -p hdr

cp ./input/hdr1.png ./hdr/input1.png
python image.py --input ./input/hdr1.png --hdr --output ./hdr/bi1.png -m bi
python image.py -l 0.02 --input ./input/hdr1.png --hdr --output ./hdr/l01.png

cp ./input/hdr2.png ./hdr/input2.png
python image.py --input ./input/hdr2.png --hdr --output ./hdr/bi2.png -m bi
python image.py -l 0.02 --input ./input/hdr2.png --hdr --output ./hdr/l02.png