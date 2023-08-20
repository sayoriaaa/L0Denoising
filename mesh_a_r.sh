root=mesh_test
mkdir -p ${root}
cd ${root}
# l=lambda a=area n=noise 0.3
mkdir -p bunny_l_16_a_n_3R
mkdir -p bunny_l_1_a_n_3R
mkdir -p bunny_l_16-1_a_n_3R

mkdir -p SharpSphere_l_16_a_n_3R
mkdir -p SharpSphere_l_1_a_n_3R
mkdir -p SharpSphere_l_16-1_a_n_3R

mkdir -p fandisk_l_16_a_n_3R
mkdir -p fandisk_l_1_a_n_3R
mkdir -p fandisk_l_16-1_a_n_3R

mkdir -p twelve_l_16_a_n_3R
mkdir -p twelve_l_1_a_n_3R
mkdir -p twelve_l_16-1_a_n_3R

mkdir -p block_l_16_a_n_3R
mkdir -p block_l_1_a_n_3R
mkdir -p block_l_16-1_a_n_3R

cd ..

model=("./input/bunny_fine.obj" "./input/SharpSphere.obj" "./input/fandisk.obj" "./input/TwelveImpulse.obj" "./input/Block.obj")

# lambda = 1/16
file=(bunny_l_16-1_a_n_3R SharpSphere_l_16-1_a_n_3R fandisk_l_16-1_a_n_3R twelve_l_16-1_a_n_3R block_l_16-1_a_n_3R)
for i in {0..4}
do
    python mesh.py --input ${model[i]} --output ./${root}/${file[i]}/area.obj -m area -n 0.3 --regulation
    cp noise.obj ./${root}/${file[i]}

    python render.py --input ${model[i]} --output ./${root}/${file[i]}/input.png
    python render.py --input ./${root}/${file[i]}/noise.obj --output ./${root}/${file[i]}/noise.png
    python render.py --input ./${root}/${file[i]}/area.obj --output ./${root}/${file[i]}/area.png
done

# lambda = 1
file=(bunny_l_1_a_n_3R SharpSphere_l_1_a_n_3R fandisk_l_1_a_n_3R twelve_l_1_a_n_3R block_l_1_a_n_3R)
for i in {0..4}
do
    python mesh.py --input ${model[i]} --output ./${root}/${file[i]}/area.obj -m area -n 0.3 -l 1 --regulation
    cp noise.obj ./${root}/${file[i]}

    python render.py --input ${model[i]} --output ./${root}/${file[i]}/input.png
    python render.py --input ./${root}/${file[i]}/noise.obj --output ./${root}/${file[i]}/noise.png
    python render.py --input ./${root}/${file[i]}/area.obj --output ./${root}/${file[i]}/area.png
done

# lambda = 16
file=(bunny_l_16_a_n_3R SharpSphere_l_16_a_n_3R fandisk_l_16_a_n_3R twelve_l_16_a_n_3R block_l_16_a_n_3R)
for i in {0..4}
do
    python mesh.py --input ${model[i]} --output ./${root}/${file[i]}/area.obj -m area -n 0.3 -l 16 --regulation
    cp noise.obj ./${root}/${file[i]}

    python render.py --input ${model[i]} --output ./${root}/${file[i]}/input.png
    python render.py --input ./${root}/${file[i]}/noise.obj --output ./${root}/${file[i]}/noise.png
    python render.py --input ./${root}/${file[i]}/area.obj --output ./${root}/${file[i]}/area.png
done



