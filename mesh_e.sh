root=mesh_test
mkdir -p ${root}
cd ${root}
# l=lambda e=edge n=noise 0.3
mkdir -p bunny_l_16_e_n_3
mkdir -p bunny_l_1_e_n_3
mkdir -p bunny_l_16-1_e_n_3

mkdir -p SharpSphere_l_16_e_n_3
mkdir -p SharpSphere_l_1_e_n_3
mkdir -p SharpSphere_l_16-1_e_n_3

mkdir -p fandisk_l_16_e_n_3
mkdir -p fandisk_l_1_e_n_3
mkdir -p fandisk_l_16-1_e_n_3

mkdir -p twelve_l_16_e_n_3
mkdir -p twelve_l_1_e_n_3
mkdir -p twelve_l_16-1_e_n_3

mkdir -p block_l_16_e_n_3
mkdir -p block_l_1_e_n_3
mkdir -p block_l_16-1_e_n_3

cd ..

model=("./input/bunny_fine.obj" "./input/SharpSphere.obj" "./input/fandisk.obj" "./input/TwelveImpulse.obj" "./input/Block.obj")

# lambda = 1/16
file=(bunny_l_16-1_e_n_3 SharpSphere_l_16-1_e_n_3 fandisk_l_16-1_e_n_3 twelve_l_16-1_e_n_3 block_l_16-1_e_n_3)
for i in {0..4}
do
    python mesh.py --input ${model[i]} --output ./${root}/${file[i]}/edge.obj -m edge -n 0.3
    cp noise.obj ./${root}/${file[i]}

    python render.py --input ${model[i]} --output ./${root}/${file[i]}/input.png
    python render.py --input ./${root}/${file[i]}/noise.obj --output ./${root}/${file[i]}/noise.png
    python render.py --input ./${root}/${file[i]}/edge.obj --output ./${root}/${file[i]}/edge.png
done

# lambda = 1
file=(bunny_l_1_e_n_3 SharpSphere_l_1_e_n_3 fandisk_l_1_e_n_3 twelve_l_1_e_n_3 block_l_1_e_n_3)
for i in {0..4}
do
    python mesh.py --input ${model[i]} --output ./${root}/${file[i]}/edge.obj -m edge -n 0.3 -l 1
    cp noise.obj ./${root}/${file[i]}

    python render.py --input ${model[i]} --output ./${root}/${file[i]}/input.png
    python render.py --input ./${root}/${file[i]}/noise.obj --output ./${root}/${file[i]}/noise.png
    python render.py --input ./${root}/${file[i]}/edge.obj --output ./${root}/${file[i]}/edge.png
done

# lambda = 16
file=(bunny_l_16_e_n_3 SharpSphere_l_16_e_n_3 fandisk_l_16_e_n_3 twelve_l_16_e_n_3 block_l_16_e_n_3)
for i in {0..4}
do
    python mesh.py --input ${model[i]} --output ./${root}/${file[i]}/edge.obj -m edge -n 0.3 -l 16
    cp noise.obj ./${root}/${file[i]}

    python render.py --input ${model[i]} --output ./${root}/${file[i]}/input.png
    python render.py --input ./${root}/${file[i]}/noise.obj --output ./${root}/${file[i]}/noise.png
    python render.py --input ./${root}/${file[i]}/edge.obj --output ./${root}/${file[i]}/edge.png
done



