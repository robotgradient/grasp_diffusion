#!/bin/bash

path_shapenet=/home/julen/data/code/str_plan/point_energies/source_data/models-OBJ/models
path_manifold=/home/julen/data/code/str_plan/point_energies/source_data/Manifold/build


cd $path_shapenet
for filename in *.obj
do
  echo $filename
  $path_manifold/manifold $filename temp.watertight.obj -s
  $path_manifold/simplify -i temp.watertight.obj -o $filename -m -r 0.02
done