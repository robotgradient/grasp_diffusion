# Generation of the Training dataset for Grasp-SE(3)DiffusionFields


We introduce the dataset generation pipeline for Grasp-SE(3)DiffusionFields.
We built our dataset through a process over Acronym dataset [1] and Shapenet-sem [2].

### Requirements
Clone https://github.com/TheCamusean/mesh_to_sdf and install
```azure
pip install -e .
```

### Dataset Generation 

1. Follow the steps in https://github.com/NVlabs/acronym to download Acronym and ShapenetSem
and simplify ShapenetSem meshes. We provide the shell script ```scripts/simplify.sh``` to simplify all the meshes in the ShapeNetSem.
**Be aware that simplifying the whole dataset might take days!!!!**
3. Organize the meshes and the Acronym grasps based on the object type with ```scripts/organize_meshes.py```.
   Set **SAVE_FOLDER** (folder in which the training dataset will be), **acronym_path** (folder where Acronym
   grasps fiels are) and **shapenet_path** (folder where the ShapeNetSem meshes are) .
3. Generate sdf for the objects with ```scripts/generate_mesh_sdf.py```. You can set for which Objects classes
you want to create the sdf selecting the elements in **OBJ_CLASSES**.



### Contact

If you have any questions or find any bugs, please let me know: [Julen Urain](http://robotgradient.com/) julen[at]robot-learning[dot]de





