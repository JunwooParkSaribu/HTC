[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13334706.svg)](https://doi.org/10.5281/zenodo.13334706)
# H2B mobility classifier
<h3>HTC classifies the mobility of H2Bs based on their trajectory image using CNN</h3> 

> [!IMPORTANT]  
> Requirements to run from source code </br>
> - C compiler
> - Python 3.10 or higher
> - TensorFlow 2.12
> - latest version of numpy
> - latest version of imageio
> - latest version of matplotlib
> - latest version of PySimpleGUI
> - latest version of [scikit-learn](https://scikit-learn.org/stable/)

> [!NOTE]  
> Binary executable(WIN) can be [downloaded](https://psilo.sorbonne-universite.fr/index.php/s/qYmL6Ywfy8kfA46)[^1]. </br>


***Input &nbsp;&nbsp; : .trxyt***<br>
***Output : Reports(.csv)*** including classified mobility(in 3 types) of H2Bs.<br><br>
This repository contains sample.trxyt file for the test of the program.<br>
Please test sample.trxyt before run the program on your own data.<br>

To run the program, open the GUI with following command.<br>
From source code:*python H2bInterface.py*

<br>
<br>

![](https://github.com/JunwooParkSaribu/HTC/blob/main/img/h2binterface_image.png)
![](https://github.com/JunwooParkSaribu/HTC/blob/main/img/cell9_image.png)


<h3> Contacts </h3>

[^1]: junwoo.park@sorbonne-universite.fr