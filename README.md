# Patching Tools for X-Ray Teacher object detection


## Install NuScenes Mini Dataset

This code assumes that the NuScenes Mini dataset is placed in the root directory of the 'CourseWork3DDetection' repository. If you don't have the dataset already, follow these steps to install it:

1. Navigate to the root directory of the 'CourseWork3DDetection' repository. You can use the following command to change the directory:

```bash
%cd /content/CourseWork3DDetection
```

2. After completing these steps, the NuScenes Mini dataset should be available in the expected directory for use in your project:

``` bash
!mkdir -p ../data/sets/nuscenes 
!wget https://www.nuscenes.org/data/v1.0-mini.tgz  
!tar -xf v1.0-mini.tgz -C ../data/sets/nuscenes  
```

