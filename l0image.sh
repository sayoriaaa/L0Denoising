mkdir -p image_test
cd image_test

mkdir -p nbu_l_1e-1
mkdir -p nbu_l_5e-2
mkdir -p nbu_l_1e-2

mkdir -p flower_l_1e-1
mkdir -p flower_l_5e-2
mkdir -p flower_l_1e-2
cd ..

# nbu
python image.py -l 0.1 --input ./input/nbu.jpg 

cp input_line.png ./image_test/nbu_l_1e-1
cp line.png ./image_test/nbu_l_1e-1
cp input_signal.png ./image_test/nbu_l_1e-1
cp signal.png ./image_test/nbu_l_1e-1
cp res.jpg ./image_test/nbu_l_1e-1

python image.py -l 0.05 --input ./input/nbu.jpg 

cp input_line.png ./image_test/nbu_l_5e-2
cp line.png ./image_test/nbu_l_5e-2
cp input_signal.png ./image_test/nbu_l_5e-2
cp signal.png ./image_test/nbu_l_5e-2
cp res.jpg ./image_test/nbu_l_5e-2

python image.py -l 0.01 --input ./input/nbu.jpg 

cp input_line.png ./image_test/nbu_l_1e-2
cp line.png ./image_test/nbu_l_1e-2
cp input_signal.png ./image_test/nbu_l_1e-2
cp signal.png ./image_test/nbu_l_1e-2
cp res.jpg ./image_test/nbu_l_1e-2

# flower
python image.py -l 0.1 --input ./input/flower.png 

cp input_line.png ./image_test/flower_l_1e-1
cp line.png ./image_test/flower_l_1e-1
cp input_signal.png ./image_test/flower_l_1e-1
cp signal.png ./image_test/flower_l_1e-1
cp res.jpg ./image_test/flower_l_1e-1

python image.py -l 0.05 --input ./input/flower.png 

cp input_line.png ./image_test/flower_l_5e-2
cp line.png ./image_test/flower_l_5e-2
cp input_signal.png ./image_test/flower_l_5e-2
cp signal.png ./image_test/flower_l_5e-2
cp res.jpg ./image_test/flower_l_5e-2

python image.py -l 0.01 --input ./input/flower.png 

cp input_line.png ./image_test/flower_l_1e-2
cp line.png ./image_test/flower_l_1e-2
cp input_signal.png ./image_test/flower_l_1e-2
cp signal.png ./image_test/flower_l_1e-2
cp res.jpg ./image_test/flower_l_1e-2