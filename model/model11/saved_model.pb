??,
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58??'
?
%Adam/v/htc/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/htc/batch_normalization_5/beta
?
9Adam/v/htc/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_5/beta*
_output_shapes
:*
dtype0
?
%Adam/m/htc/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/htc/batch_normalization_5/beta
?
9Adam/m/htc/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_5/beta*
_output_shapes
:*
dtype0
?
&Adam/v/htc/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/htc/batch_normalization_5/gamma
?
:Adam/v/htc/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
?
&Adam/m/htc/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/htc/batch_normalization_5/gamma
?
:Adam/m/htc/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
?
Adam/v/htc/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/htc/dense/bias
{
)Adam/v/htc/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/dense/bias*
_output_shapes
:*
dtype0
?
Adam/m/htc/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/htc/dense/bias
{
)Adam/m/htc/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/dense/bias*
_output_shapes
:*
dtype0
?
Adam/v/htc/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$*(
shared_nameAdam/v/htc/dense/kernel
?
+Adam/v/htc/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/dense/kernel*
_output_shapes
:	?$*
dtype0
?
Adam/m/htc/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$*(
shared_nameAdam/m/htc/dense/kernel
?
+Adam/m/htc/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/dense/kernel*
_output_shapes
:	?$*
dtype0
?
%Adam/v/htc/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/v/htc/batch_normalization_4/beta
?
9Adam/v/htc/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_4/beta*
_output_shapes	
:?*
dtype0
?
%Adam/m/htc/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/m/htc/batch_normalization_4/beta
?
9Adam/m/htc/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_4/beta*
_output_shapes	
:?*
dtype0
?
&Adam/v/htc/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/v/htc/batch_normalization_4/gamma
?
:Adam/v/htc/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_4/gamma*
_output_shapes	
:?*
dtype0
?
&Adam/m/htc/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/m/htc/batch_normalization_4/gamma
?
:Adam/m/htc/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_4/gamma*
_output_shapes	
:?*
dtype0
?
Adam/v/htc/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/v/htc/conv2d_4/bias
?
,Adam/v/htc/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_4/bias*
_output_shapes	
:?*
dtype0
?
Adam/m/htc/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/m/htc/conv2d_4/bias
?
,Adam/m/htc/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_4/bias*
_output_shapes	
:?*
dtype0
?
Adam/v/htc/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameAdam/v/htc/conv2d_4/kernel
?
.Adam/v/htc/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_4/kernel*(
_output_shapes
:??*
dtype0
?
Adam/m/htc/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameAdam/m/htc/conv2d_4/kernel
?
.Adam/m/htc/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_4/kernel*(
_output_shapes
:??*
dtype0
?
%Adam/v/htc/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/v/htc/batch_normalization_3/beta
?
9Adam/v/htc/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_3/beta*
_output_shapes	
:?*
dtype0
?
%Adam/m/htc/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/m/htc/batch_normalization_3/beta
?
9Adam/m/htc/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_3/beta*
_output_shapes	
:?*
dtype0
?
&Adam/v/htc/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/v/htc/batch_normalization_3/gamma
?
:Adam/v/htc/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_3/gamma*
_output_shapes	
:?*
dtype0
?
&Adam/m/htc/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/m/htc/batch_normalization_3/gamma
?
:Adam/m/htc/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_3/gamma*
_output_shapes	
:?*
dtype0
?
Adam/v/htc/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/v/htc/conv2d_3/bias
?
,Adam/v/htc/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_3/bias*
_output_shapes	
:?*
dtype0
?
Adam/m/htc/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/m/htc/conv2d_3/bias
?
,Adam/m/htc/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_3/bias*
_output_shapes	
:?*
dtype0
?
Adam/v/htc/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameAdam/v/htc/conv2d_3/kernel
?
.Adam/v/htc/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_3/kernel*(
_output_shapes
:??*
dtype0
?
Adam/m/htc/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameAdam/m/htc/conv2d_3/kernel
?
.Adam/m/htc/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_3/kernel*(
_output_shapes
:??*
dtype0
?
%Adam/v/htc/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/v/htc/batch_normalization_2/beta
?
9Adam/v/htc/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_2/beta*
_output_shapes	
:?*
dtype0
?
%Adam/m/htc/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/m/htc/batch_normalization_2/beta
?
9Adam/m/htc/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_2/beta*
_output_shapes	
:?*
dtype0
?
&Adam/v/htc/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/v/htc/batch_normalization_2/gamma
?
:Adam/v/htc/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_2/gamma*
_output_shapes	
:?*
dtype0
?
&Adam/m/htc/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/m/htc/batch_normalization_2/gamma
?
:Adam/m/htc/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_2/gamma*
_output_shapes	
:?*
dtype0
?
Adam/v/htc/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/v/htc/conv2d_2/bias
?
,Adam/v/htc/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
Adam/m/htc/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/m/htc/conv2d_2/bias
?
,Adam/m/htc/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
Adam/v/htc/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameAdam/v/htc/conv2d_2/kernel
?
.Adam/v/htc/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_2/kernel*'
_output_shapes
:@?*
dtype0
?
Adam/m/htc/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameAdam/m/htc/conv2d_2/kernel
?
.Adam/m/htc/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_2/kernel*'
_output_shapes
:@?*
dtype0
?
%Adam/v/htc/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/v/htc/batch_normalization_1/beta
?
9Adam/v/htc/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
%Adam/m/htc/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/m/htc/batch_normalization_1/beta
?
9Adam/m/htc/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
&Adam/v/htc/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/v/htc/batch_normalization_1/gamma
?
:Adam/v/htc/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
&Adam/m/htc/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/m/htc/batch_normalization_1/gamma
?
:Adam/m/htc/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
Adam/v/htc/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/v/htc/conv2d_1/bias
?
,Adam/v/htc/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_1/bias*
_output_shapes
:@*
dtype0
?
Adam/m/htc/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/m/htc/conv2d_1/bias
?
,Adam/m/htc/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_1/bias*
_output_shapes
:@*
dtype0
?
Adam/v/htc/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/v/htc/conv2d_1/kernel
?
.Adam/v/htc/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
?
Adam/m/htc/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/m/htc/conv2d_1/kernel
?
.Adam/m/htc/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
?
#Adam/v/htc/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/v/htc/batch_normalization/beta
?
7Adam/v/htc/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#Adam/v/htc/batch_normalization/beta*
_output_shapes
: *
dtype0
?
#Adam/m/htc/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/m/htc/batch_normalization/beta
?
7Adam/m/htc/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#Adam/m/htc/batch_normalization/beta*
_output_shapes
: *
dtype0
?
$Adam/v/htc/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/v/htc/batch_normalization/gamma
?
8Adam/v/htc/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$Adam/v/htc/batch_normalization/gamma*
_output_shapes
: *
dtype0
?
$Adam/m/htc/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/m/htc/batch_normalization/gamma
?
8Adam/m/htc/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$Adam/m/htc/batch_normalization/gamma*
_output_shapes
: *
dtype0
?
Adam/v/htc/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/htc/conv2d/bias
}
*Adam/v/htc/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d/bias*
_output_shapes
: *
dtype0
?
Adam/m/htc/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/htc/conv2d/bias
}
*Adam/m/htc/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d/bias*
_output_shapes
: *
dtype0
?
Adam/v/htc/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/v/htc/conv2d/kernel
?
,Adam/v/htc/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d/kernel*&
_output_shapes
: *
dtype0
?
Adam/m/htc/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/m/htc/conv2d/kernel
?
,Adam/m/htc/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
?
)htc/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)htc/batch_normalization_5/moving_variance
?
=htc/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
?
%htc/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%htc/batch_normalization_5/moving_mean
?
9htc/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
?
htc/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name htc/batch_normalization_5/beta
?
2htc/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_5/beta*
_output_shapes
:*
dtype0
?
htc/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!htc/batch_normalization_5/gamma
?
3htc/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
t
htc/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehtc/dense/bias
m
"htc/dense/bias/Read/ReadVariableOpReadVariableOphtc/dense/bias*
_output_shapes
:*
dtype0
}
htc/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$*!
shared_namehtc/dense/kernel
v
$htc/dense/kernel/Read/ReadVariableOpReadVariableOphtc/dense/kernel*
_output_shapes
:	?$*
dtype0
?
)htc/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)htc/batch_normalization_4/moving_variance
?
=htc/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_4/moving_variance*
_output_shapes	
:?*
dtype0
?
%htc/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%htc/batch_normalization_4/moving_mean
?
9htc/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_4/moving_mean*
_output_shapes	
:?*
dtype0
?
htc/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name htc/batch_normalization_4/beta
?
2htc/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_4/beta*
_output_shapes	
:?*
dtype0
?
htc/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!htc/batch_normalization_4/gamma
?
3htc/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_4/gamma*
_output_shapes	
:?*
dtype0
{
htc/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehtc/conv2d_4/bias
t
%htc/conv2d_4/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_4/bias*
_output_shapes	
:?*
dtype0
?
htc/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_namehtc/conv2d_4/kernel
?
'htc/conv2d_4/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_4/kernel*(
_output_shapes
:??*
dtype0
?
)htc/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)htc/batch_normalization_3/moving_variance
?
=htc/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_3/moving_variance*
_output_shapes	
:?*
dtype0
?
%htc/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%htc/batch_normalization_3/moving_mean
?
9htc/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_3/moving_mean*
_output_shapes	
:?*
dtype0
?
htc/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name htc/batch_normalization_3/beta
?
2htc/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_3/beta*
_output_shapes	
:?*
dtype0
?
htc/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!htc/batch_normalization_3/gamma
?
3htc/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_3/gamma*
_output_shapes	
:?*
dtype0
{
htc/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehtc/conv2d_3/bias
t
%htc/conv2d_3/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_3/bias*
_output_shapes	
:?*
dtype0
?
htc/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_namehtc/conv2d_3/kernel
?
'htc/conv2d_3/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_3/kernel*(
_output_shapes
:??*
dtype0
?
)htc/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)htc/batch_normalization_2/moving_variance
?
=htc/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_2/moving_variance*
_output_shapes	
:?*
dtype0
?
%htc/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%htc/batch_normalization_2/moving_mean
?
9htc/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_2/moving_mean*
_output_shapes	
:?*
dtype0
?
htc/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name htc/batch_normalization_2/beta
?
2htc/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_2/beta*
_output_shapes	
:?*
dtype0
?
htc/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!htc/batch_normalization_2/gamma
?
3htc/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_2/gamma*
_output_shapes	
:?*
dtype0
{
htc/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehtc/conv2d_2/bias
t
%htc/conv2d_2/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
htc/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_namehtc/conv2d_2/kernel
?
'htc/conv2d_2/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_2/kernel*'
_output_shapes
:@?*
dtype0
?
)htc/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)htc/batch_normalization_1/moving_variance
?
=htc/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
?
%htc/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%htc/batch_normalization_1/moving_mean
?
9htc/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
htc/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name htc/batch_normalization_1/beta
?
2htc/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
htc/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!htc/batch_normalization_1/gamma
?
3htc/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
z
htc/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namehtc/conv2d_1/bias
s
%htc/conv2d_1/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_1/bias*
_output_shapes
:@*
dtype0
?
htc/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_namehtc/conv2d_1/kernel
?
'htc/conv2d_1/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
?
'htc/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'htc/batch_normalization/moving_variance
?
;htc/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp'htc/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
?
#htc/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#htc/batch_normalization/moving_mean
?
7htc/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp#htc/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
?
htc/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namehtc/batch_normalization/beta
?
0htc/batch_normalization/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization/beta*
_output_shapes
: *
dtype0
?
htc/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namehtc/batch_normalization/gamma
?
1htc/batch_normalization/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization/gamma*
_output_shapes
: *
dtype0
v
htc/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namehtc/conv2d/bias
o
#htc/conv2d/bias/Read/ReadVariableOpReadVariableOphtc/conv2d/bias*
_output_shapes
: *
dtype0
?
htc/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namehtc/conv2d/kernel

%htc/conv2d/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d/kernel*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1htc/conv2d/kernelhtc/conv2d/biashtc/batch_normalization/gammahtc/batch_normalization/beta#htc/batch_normalization/moving_mean'htc/batch_normalization/moving_variancehtc/conv2d_1/kernelhtc/conv2d_1/biashtc/batch_normalization_1/gammahtc/batch_normalization_1/beta%htc/batch_normalization_1/moving_mean)htc/batch_normalization_1/moving_variancehtc/conv2d_2/kernelhtc/conv2d_2/biashtc/batch_normalization_2/gammahtc/batch_normalization_2/beta%htc/batch_normalization_2/moving_mean)htc/batch_normalization_2/moving_variancehtc/conv2d_3/kernelhtc/conv2d_3/biashtc/batch_normalization_3/gammahtc/batch_normalization_3/beta%htc/batch_normalization_3/moving_mean)htc/batch_normalization_3/moving_variancehtc/conv2d_4/kernelhtc/conv2d_4/biashtc/batch_normalization_4/gammahtc/batch_normalization_4/beta%htc/batch_normalization_4/moving_mean)htc/batch_normalization_4/moving_variancehtc/dense/kernelhtc/dense/bias%htc/batch_normalization_5/moving_mean)htc/batch_normalization_5/moving_variancehtc/batch_normalization_5/betahtc/batch_normalization_5/gamma*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference_signature_wrapper_47639

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	
train_loss

train_accuracy
	test_loss
test_accuracy
	conv1
	pool1

batch1
relu_activ1
	conv2
	pool2

batch2
relu_activ2
	conv3
	pool3

batch3
relu_activ3
	conv4
	pool4

batch4
relu_activ4
	conv5
	pool5

batch5
 relu_activ5
!dropout1
"flatten
	#d_fin
$	batch_fin
%
soft_activ
&	test_step
'
train_step
(
signatures*
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29
G30
H31
I32
J33
K34
L35
M36
N37
O38
P39
Q40
R41
S42
T43*
?
10
21
32
43
74
85
96
:7
=8
>9
?10
@11
C12
D13
E14
F15
I16
J17
K18
L19
O20
P21
Q22
R23*
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ztrace_0
[trace_1
\trace_2
]trace_3* 
6
^trace_0
_trace_1
`trace_2
atrace_3* 
* 
?
b
_variables
c_iterations
d_learning_rate
e_index_dict
f
_momentums
g_velocities
h_update_step_xla*
8
i	variables
j	keras_api
	)total
	*count*
H
k	variables
l	keras_api
	+total
	,count
m
_fn_kwargs*
8
n	variables
o	keras_api
	-total
	.count*
H
p	variables
q	keras_api
	/total
	0count
r
_fn_kwargs*
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

1kernel
2bias
 y_jit_compiled_convolution_op*
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	3gamma
4beta
5moving_mean
6moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

7kernel
8bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	9gamma
:beta
;moving_mean
<moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

=kernel
>bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	?gamma
@beta
Amoving_mean
Bmoving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Ckernel
Dbias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Ikernel
Jbias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Okernel
Pbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0* 

?trace_0* 

?serving_default* 
GA
VARIABLE_VALUEtotal_3&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEcount_3&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEtotal_2&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEcount_2&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEtotal_1&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEcount_1&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
E?
VARIABLE_VALUEtotal&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
E?
VARIABLE_VALUEcount&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEhtc/conv2d/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEhtc/conv2d/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEhtc/batch_normalization/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEhtc/batch_normalization/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#htc/batch_normalization/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'htc/batch_normalization/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEhtc/conv2d_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEhtc/conv2d_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEhtc/batch_normalization_1/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhtc/batch_normalization_1/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%htc/batch_normalization_1/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)htc/batch_normalization_1/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEhtc/conv2d_2/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEhtc/conv2d_2/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEhtc/batch_normalization_2/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhtc/batch_normalization_2/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%htc/batch_normalization_2/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)htc/batch_normalization_2/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEhtc/conv2d_3/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEhtc/conv2d_3/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEhtc/batch_normalization_3/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhtc/batch_normalization_3/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%htc/batch_normalization_3/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)htc/batch_normalization_3/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEhtc/conv2d_4/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEhtc/conv2d_4/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEhtc/batch_normalization_4/gamma'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhtc/batch_normalization_4/beta'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%htc/batch_normalization_4/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)htc/batch_normalization_4/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEhtc/dense/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEhtc/dense/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEhtc/batch_normalization_5/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhtc/batch_normalization_5/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%htc/batch_normalization_5/moving_mean'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)htc/batch_normalization_5/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
?
)0
*1
+2
,3
-4
.5
/6
07
58
69
;10
<11
A12
B13
G14
H15
M16
N17
S18
T19*
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23
%24*
 
	0

1
2
3*
* 
J
	
train_loss

train_accuracy
	test_loss
test_accuracy*
* 
* 
* 
* 
* 
* 
* 
* 
?
c0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23*
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23*
* 

)0
*1*

i	variables*

+0
,1*

k	variables*
* 

-0
.1*

n	variables*

/0
01*

p	variables*
* 

10
21*

10
21*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
30
41
52
63*

30
41*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

70
81*

70
81*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
90
:1
;2
<3*

90
:1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

=0
>1*

=0
>1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
?0
@1
A2
B3*

?0
@1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

C0
D1*

C0
D1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
E0
F1
G2
H3*

E0
F1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

I0
J1*

I0
J1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
K0
L1
M2
N3*

K0
L1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

O0
P1*

O0
P1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
 
Q0
R1
S2
T3*

Q0
R1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
c]
VARIABLE_VALUEAdam/m/htc/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/htc/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/htc/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/htc/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/htc/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/htc/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/htc/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/htc/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/htc/conv2d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/htc/conv2d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/htc/conv2d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/htc/conv2d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/htc/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/htc/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/htc/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/htc/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/htc/conv2d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/htc/conv2d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/htc/conv2d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/htc/conv2d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/htc/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/htc/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/htc/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/htc/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/htc/conv2d_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/htc/conv2d_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/htc/conv2d_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/htc/conv2d_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/htc/batch_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/htc/batch_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/htc/batch_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/htc/batch_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/htc/conv2d_4/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/htc/conv2d_4/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/htc/conv2d_4/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/htc/conv2d_4/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/htc/batch_normalization_4/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/htc/batch_normalization_4/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/htc/batch_normalization_4/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/htc/batch_normalization_4/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/htc/dense/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/htc/dense/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/htc/dense/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/htc/dense/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/htc/batch_normalization_5/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/htc/batch_normalization_5/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/htc/batch_normalization_5/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/htc/batch_normalization_5/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

;0
<1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

A0
B1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

S0
T1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp%htc/conv2d/kernel/Read/ReadVariableOp#htc/conv2d/bias/Read/ReadVariableOp1htc/batch_normalization/gamma/Read/ReadVariableOp0htc/batch_normalization/beta/Read/ReadVariableOp7htc/batch_normalization/moving_mean/Read/ReadVariableOp;htc/batch_normalization/moving_variance/Read/ReadVariableOp'htc/conv2d_1/kernel/Read/ReadVariableOp%htc/conv2d_1/bias/Read/ReadVariableOp3htc/batch_normalization_1/gamma/Read/ReadVariableOp2htc/batch_normalization_1/beta/Read/ReadVariableOp9htc/batch_normalization_1/moving_mean/Read/ReadVariableOp=htc/batch_normalization_1/moving_variance/Read/ReadVariableOp'htc/conv2d_2/kernel/Read/ReadVariableOp%htc/conv2d_2/bias/Read/ReadVariableOp3htc/batch_normalization_2/gamma/Read/ReadVariableOp2htc/batch_normalization_2/beta/Read/ReadVariableOp9htc/batch_normalization_2/moving_mean/Read/ReadVariableOp=htc/batch_normalization_2/moving_variance/Read/ReadVariableOp'htc/conv2d_3/kernel/Read/ReadVariableOp%htc/conv2d_3/bias/Read/ReadVariableOp3htc/batch_normalization_3/gamma/Read/ReadVariableOp2htc/batch_normalization_3/beta/Read/ReadVariableOp9htc/batch_normalization_3/moving_mean/Read/ReadVariableOp=htc/batch_normalization_3/moving_variance/Read/ReadVariableOp'htc/conv2d_4/kernel/Read/ReadVariableOp%htc/conv2d_4/bias/Read/ReadVariableOp3htc/batch_normalization_4/gamma/Read/ReadVariableOp2htc/batch_normalization_4/beta/Read/ReadVariableOp9htc/batch_normalization_4/moving_mean/Read/ReadVariableOp=htc/batch_normalization_4/moving_variance/Read/ReadVariableOp$htc/dense/kernel/Read/ReadVariableOp"htc/dense/bias/Read/ReadVariableOp3htc/batch_normalization_5/gamma/Read/ReadVariableOp2htc/batch_normalization_5/beta/Read/ReadVariableOp9htc/batch_normalization_5/moving_mean/Read/ReadVariableOp=htc/batch_normalization_5/moving_variance/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp,Adam/m/htc/conv2d/kernel/Read/ReadVariableOp,Adam/v/htc/conv2d/kernel/Read/ReadVariableOp*Adam/m/htc/conv2d/bias/Read/ReadVariableOp*Adam/v/htc/conv2d/bias/Read/ReadVariableOp8Adam/m/htc/batch_normalization/gamma/Read/ReadVariableOp8Adam/v/htc/batch_normalization/gamma/Read/ReadVariableOp7Adam/m/htc/batch_normalization/beta/Read/ReadVariableOp7Adam/v/htc/batch_normalization/beta/Read/ReadVariableOp.Adam/m/htc/conv2d_1/kernel/Read/ReadVariableOp.Adam/v/htc/conv2d_1/kernel/Read/ReadVariableOp,Adam/m/htc/conv2d_1/bias/Read/ReadVariableOp,Adam/v/htc/conv2d_1/bias/Read/ReadVariableOp:Adam/m/htc/batch_normalization_1/gamma/Read/ReadVariableOp:Adam/v/htc/batch_normalization_1/gamma/Read/ReadVariableOp9Adam/m/htc/batch_normalization_1/beta/Read/ReadVariableOp9Adam/v/htc/batch_normalization_1/beta/Read/ReadVariableOp.Adam/m/htc/conv2d_2/kernel/Read/ReadVariableOp.Adam/v/htc/conv2d_2/kernel/Read/ReadVariableOp,Adam/m/htc/conv2d_2/bias/Read/ReadVariableOp,Adam/v/htc/conv2d_2/bias/Read/ReadVariableOp:Adam/m/htc/batch_normalization_2/gamma/Read/ReadVariableOp:Adam/v/htc/batch_normalization_2/gamma/Read/ReadVariableOp9Adam/m/htc/batch_normalization_2/beta/Read/ReadVariableOp9Adam/v/htc/batch_normalization_2/beta/Read/ReadVariableOp.Adam/m/htc/conv2d_3/kernel/Read/ReadVariableOp.Adam/v/htc/conv2d_3/kernel/Read/ReadVariableOp,Adam/m/htc/conv2d_3/bias/Read/ReadVariableOp,Adam/v/htc/conv2d_3/bias/Read/ReadVariableOp:Adam/m/htc/batch_normalization_3/gamma/Read/ReadVariableOp:Adam/v/htc/batch_normalization_3/gamma/Read/ReadVariableOp9Adam/m/htc/batch_normalization_3/beta/Read/ReadVariableOp9Adam/v/htc/batch_normalization_3/beta/Read/ReadVariableOp.Adam/m/htc/conv2d_4/kernel/Read/ReadVariableOp.Adam/v/htc/conv2d_4/kernel/Read/ReadVariableOp,Adam/m/htc/conv2d_4/bias/Read/ReadVariableOp,Adam/v/htc/conv2d_4/bias/Read/ReadVariableOp:Adam/m/htc/batch_normalization_4/gamma/Read/ReadVariableOp:Adam/v/htc/batch_normalization_4/gamma/Read/ReadVariableOp9Adam/m/htc/batch_normalization_4/beta/Read/ReadVariableOp9Adam/v/htc/batch_normalization_4/beta/Read/ReadVariableOp+Adam/m/htc/dense/kernel/Read/ReadVariableOp+Adam/v/htc/dense/kernel/Read/ReadVariableOp)Adam/m/htc/dense/bias/Read/ReadVariableOp)Adam/v/htc/dense/bias/Read/ReadVariableOp:Adam/m/htc/batch_normalization_5/gamma/Read/ReadVariableOp:Adam/v/htc/batch_normalization_5/gamma/Read/ReadVariableOp9Adam/m/htc/batch_normalization_5/beta/Read/ReadVariableOp9Adam/v/htc/batch_normalization_5/beta/Read/ReadVariableOpConst*k
Tind
b2`	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *'
f"R 
__inference__traced_save_49210
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotal_3count_3total_2count_2total_1count_1totalcounthtc/conv2d/kernelhtc/conv2d/biashtc/batch_normalization/gammahtc/batch_normalization/beta#htc/batch_normalization/moving_mean'htc/batch_normalization/moving_variancehtc/conv2d_1/kernelhtc/conv2d_1/biashtc/batch_normalization_1/gammahtc/batch_normalization_1/beta%htc/batch_normalization_1/moving_mean)htc/batch_normalization_1/moving_variancehtc/conv2d_2/kernelhtc/conv2d_2/biashtc/batch_normalization_2/gammahtc/batch_normalization_2/beta%htc/batch_normalization_2/moving_mean)htc/batch_normalization_2/moving_variancehtc/conv2d_3/kernelhtc/conv2d_3/biashtc/batch_normalization_3/gammahtc/batch_normalization_3/beta%htc/batch_normalization_3/moving_mean)htc/batch_normalization_3/moving_variancehtc/conv2d_4/kernelhtc/conv2d_4/biashtc/batch_normalization_4/gammahtc/batch_normalization_4/beta%htc/batch_normalization_4/moving_mean)htc/batch_normalization_4/moving_variancehtc/dense/kernelhtc/dense/biashtc/batch_normalization_5/gammahtc/batch_normalization_5/beta%htc/batch_normalization_5/moving_mean)htc/batch_normalization_5/moving_variance	iterationlearning_rateAdam/m/htc/conv2d/kernelAdam/v/htc/conv2d/kernelAdam/m/htc/conv2d/biasAdam/v/htc/conv2d/bias$Adam/m/htc/batch_normalization/gamma$Adam/v/htc/batch_normalization/gamma#Adam/m/htc/batch_normalization/beta#Adam/v/htc/batch_normalization/betaAdam/m/htc/conv2d_1/kernelAdam/v/htc/conv2d_1/kernelAdam/m/htc/conv2d_1/biasAdam/v/htc/conv2d_1/bias&Adam/m/htc/batch_normalization_1/gamma&Adam/v/htc/batch_normalization_1/gamma%Adam/m/htc/batch_normalization_1/beta%Adam/v/htc/batch_normalization_1/betaAdam/m/htc/conv2d_2/kernelAdam/v/htc/conv2d_2/kernelAdam/m/htc/conv2d_2/biasAdam/v/htc/conv2d_2/bias&Adam/m/htc/batch_normalization_2/gamma&Adam/v/htc/batch_normalization_2/gamma%Adam/m/htc/batch_normalization_2/beta%Adam/v/htc/batch_normalization_2/betaAdam/m/htc/conv2d_3/kernelAdam/v/htc/conv2d_3/kernelAdam/m/htc/conv2d_3/biasAdam/v/htc/conv2d_3/bias&Adam/m/htc/batch_normalization_3/gamma&Adam/v/htc/batch_normalization_3/gamma%Adam/m/htc/batch_normalization_3/beta%Adam/v/htc/batch_normalization_3/betaAdam/m/htc/conv2d_4/kernelAdam/v/htc/conv2d_4/kernelAdam/m/htc/conv2d_4/biasAdam/v/htc/conv2d_4/bias&Adam/m/htc/batch_normalization_4/gamma&Adam/v/htc/batch_normalization_4/gamma%Adam/m/htc/batch_normalization_4/beta%Adam/v/htc/batch_normalization_4/betaAdam/m/htc/dense/kernelAdam/v/htc/dense/kernelAdam/m/htc/dense/biasAdam/v/htc/dense/bias&Adam/m/htc/batch_normalization_5/gamma&Adam/v/htc/batch_normalization_5/gamma%Adam/m/htc/batch_normalization_5/beta%Adam/v/htc/batch_normalization_5/beta*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__traced_restore_49502??#
? 
?
"__inference__update_step_xla_46487
gradient
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource: +
sub_3_readvariableop_resource: ??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
: *
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
: n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
: ?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
: L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
: O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
: f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ա
?
>__inference_htc_layer_call_and_return_conditional_losses_47938
x?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_2_conv2d_readvariableop_resource:@?7
(conv2d_2_biasadd_readvariableop_resource:	?<
-batch_normalization_2_readvariableop_resource:	?>
/batch_normalization_2_readvariableop_1_resource:	?M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_3_conv2d_readvariableop_resource:??7
(conv2d_3_biasadd_readvariableop_resource:	?<
-batch_normalization_3_readvariableop_resource:	?>
/batch_normalization_3_readvariableop_1_resource:	?M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_4_conv2d_readvariableop_resource:??7
(conv2d_4_biasadd_readvariableop_resource:	?<
-batch_normalization_4_readvariableop_resource:	?>
/batch_normalization_4_readvariableop_1_resource:	?M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	?7
$dense_matmul_readvariableop_resource:	?$3
%dense_biasadd_readvariableop_resource:@
2batch_normalization_5_cast_readvariableop_resource:B
4batch_normalization_5_cast_1_readvariableop_resource:B
4batch_normalization_5_cast_2_readvariableop_resource:B
4batch_normalization_5_cast_3_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?)batch_normalization_5/Cast/ReadVariableOp?+batch_normalization_5/Cast_1/ReadVariableOp?+batch_normalization_5/Cast_2/ReadVariableOp?+batch_normalization_5/Cast_3/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:?????????bb *
ksize
*
paddingVALID*
strides
?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????bb : : : : :*
epsilon%o?:*
is_training( v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????bb ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( {
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:???????????
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( {
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:???????????
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_4/Conv2DConv2Dre_lu_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( {
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transposere_lu_4/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:??????????f
dropout/IdentityIdentitytranspose:y:0*
T0*0
_output_shapes
:??????????i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_1	Transposedropout/Identity:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   v
flatten/ReshapeReshapetranspose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????$?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_5/batchnorm/addAddV23batch_normalization_5/Cast_1/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:?
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_5/batchnorm/mul_1Muldense/Softmax:softmax:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
%batch_normalization_5/batchnorm/mul_2Mul1batch_normalization_5/Cast/ReadVariableOp:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:?
#batch_normalization_5/batchnorm/subSub3batch_normalization_5/Cast_2/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????z
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????k
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp,^batch_normalization_5/Cast_2/ReadVariableOp,^batch_normalization_5/Cast_3/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12V
)batch_normalization_5/Cast/ReadVariableOp)batch_normalization_5/Cast/ReadVariableOp2Z
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2Z
+batch_normalization_5/Cast_2/ReadVariableOp+batch_normalization_5/Cast_2/ReadVariableOp2Z
+batch_normalization_5/Cast_3/ReadVariableOp+batch_normalization_5/Cast_3/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48177

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_48667

inputs1
matmul_readvariableop_resource:	?$-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44434

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?@
!__inference__traced_restore_49502
file_prefix"
assignvariableop_total_3: $
assignvariableop_1_count_3: $
assignvariableop_2_total_2: $
assignvariableop_3_count_2: $
assignvariableop_4_total_1: $
assignvariableop_5_count_1: "
assignvariableop_6_total: "
assignvariableop_7_count: >
$assignvariableop_8_htc_conv2d_kernel: 0
"assignvariableop_9_htc_conv2d_bias: ?
1assignvariableop_10_htc_batch_normalization_gamma: >
0assignvariableop_11_htc_batch_normalization_beta: E
7assignvariableop_12_htc_batch_normalization_moving_mean: I
;assignvariableop_13_htc_batch_normalization_moving_variance: A
'assignvariableop_14_htc_conv2d_1_kernel: @3
%assignvariableop_15_htc_conv2d_1_bias:@A
3assignvariableop_16_htc_batch_normalization_1_gamma:@@
2assignvariableop_17_htc_batch_normalization_1_beta:@G
9assignvariableop_18_htc_batch_normalization_1_moving_mean:@K
=assignvariableop_19_htc_batch_normalization_1_moving_variance:@B
'assignvariableop_20_htc_conv2d_2_kernel:@?4
%assignvariableop_21_htc_conv2d_2_bias:	?B
3assignvariableop_22_htc_batch_normalization_2_gamma:	?A
2assignvariableop_23_htc_batch_normalization_2_beta:	?H
9assignvariableop_24_htc_batch_normalization_2_moving_mean:	?L
=assignvariableop_25_htc_batch_normalization_2_moving_variance:	?C
'assignvariableop_26_htc_conv2d_3_kernel:??4
%assignvariableop_27_htc_conv2d_3_bias:	?B
3assignvariableop_28_htc_batch_normalization_3_gamma:	?A
2assignvariableop_29_htc_batch_normalization_3_beta:	?H
9assignvariableop_30_htc_batch_normalization_3_moving_mean:	?L
=assignvariableop_31_htc_batch_normalization_3_moving_variance:	?C
'assignvariableop_32_htc_conv2d_4_kernel:??4
%assignvariableop_33_htc_conv2d_4_bias:	?B
3assignvariableop_34_htc_batch_normalization_4_gamma:	?A
2assignvariableop_35_htc_batch_normalization_4_beta:	?H
9assignvariableop_36_htc_batch_normalization_4_moving_mean:	?L
=assignvariableop_37_htc_batch_normalization_4_moving_variance:	?7
$assignvariableop_38_htc_dense_kernel:	?$0
"assignvariableop_39_htc_dense_bias:A
3assignvariableop_40_htc_batch_normalization_5_gamma:@
2assignvariableop_41_htc_batch_normalization_5_beta:G
9assignvariableop_42_htc_batch_normalization_5_moving_mean:K
=assignvariableop_43_htc_batch_normalization_5_moving_variance:'
assignvariableop_44_iteration:	 +
!assignvariableop_45_learning_rate: F
,assignvariableop_46_adam_m_htc_conv2d_kernel: F
,assignvariableop_47_adam_v_htc_conv2d_kernel: 8
*assignvariableop_48_adam_m_htc_conv2d_bias: 8
*assignvariableop_49_adam_v_htc_conv2d_bias: F
8assignvariableop_50_adam_m_htc_batch_normalization_gamma: F
8assignvariableop_51_adam_v_htc_batch_normalization_gamma: E
7assignvariableop_52_adam_m_htc_batch_normalization_beta: E
7assignvariableop_53_adam_v_htc_batch_normalization_beta: H
.assignvariableop_54_adam_m_htc_conv2d_1_kernel: @H
.assignvariableop_55_adam_v_htc_conv2d_1_kernel: @:
,assignvariableop_56_adam_m_htc_conv2d_1_bias:@:
,assignvariableop_57_adam_v_htc_conv2d_1_bias:@H
:assignvariableop_58_adam_m_htc_batch_normalization_1_gamma:@H
:assignvariableop_59_adam_v_htc_batch_normalization_1_gamma:@G
9assignvariableop_60_adam_m_htc_batch_normalization_1_beta:@G
9assignvariableop_61_adam_v_htc_batch_normalization_1_beta:@I
.assignvariableop_62_adam_m_htc_conv2d_2_kernel:@?I
.assignvariableop_63_adam_v_htc_conv2d_2_kernel:@?;
,assignvariableop_64_adam_m_htc_conv2d_2_bias:	?;
,assignvariableop_65_adam_v_htc_conv2d_2_bias:	?I
:assignvariableop_66_adam_m_htc_batch_normalization_2_gamma:	?I
:assignvariableop_67_adam_v_htc_batch_normalization_2_gamma:	?H
9assignvariableop_68_adam_m_htc_batch_normalization_2_beta:	?H
9assignvariableop_69_adam_v_htc_batch_normalization_2_beta:	?J
.assignvariableop_70_adam_m_htc_conv2d_3_kernel:??J
.assignvariableop_71_adam_v_htc_conv2d_3_kernel:??;
,assignvariableop_72_adam_m_htc_conv2d_3_bias:	?;
,assignvariableop_73_adam_v_htc_conv2d_3_bias:	?I
:assignvariableop_74_adam_m_htc_batch_normalization_3_gamma:	?I
:assignvariableop_75_adam_v_htc_batch_normalization_3_gamma:	?H
9assignvariableop_76_adam_m_htc_batch_normalization_3_beta:	?H
9assignvariableop_77_adam_v_htc_batch_normalization_3_beta:	?J
.assignvariableop_78_adam_m_htc_conv2d_4_kernel:??J
.assignvariableop_79_adam_v_htc_conv2d_4_kernel:??;
,assignvariableop_80_adam_m_htc_conv2d_4_bias:	?;
,assignvariableop_81_adam_v_htc_conv2d_4_bias:	?I
:assignvariableop_82_adam_m_htc_batch_normalization_4_gamma:	?I
:assignvariableop_83_adam_v_htc_batch_normalization_4_gamma:	?H
9assignvariableop_84_adam_m_htc_batch_normalization_4_beta:	?H
9assignvariableop_85_adam_v_htc_batch_normalization_4_beta:	?>
+assignvariableop_86_adam_m_htc_dense_kernel:	?$>
+assignvariableop_87_adam_v_htc_dense_kernel:	?$7
)assignvariableop_88_adam_m_htc_dense_bias:7
)assignvariableop_89_adam_v_htc_dense_bias:H
:assignvariableop_90_adam_m_htc_batch_normalization_5_gamma:H
:assignvariableop_91_adam_v_htc_batch_normalization_5_gamma:G
9assignvariableop_92_adam_m_htc_batch_normalization_5_beta:G
9assignvariableop_93_adam_v_htc_batch_normalization_5_beta:
identity_95??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*?"
value?"B?"_B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*?
value?B?_B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*m
dtypesc
a2_	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_total_3Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_count_3Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_total_2Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_count_2Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_total_1Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_htc_conv2d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_htc_conv2d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp1assignvariableop_10_htc_batch_normalization_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_htc_batch_normalization_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_htc_batch_normalization_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp;assignvariableop_13_htc_batch_normalization_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_htc_conv2d_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_htc_conv2d_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp3assignvariableop_16_htc_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_htc_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp9assignvariableop_18_htc_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp=assignvariableop_19_htc_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_htc_conv2d_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_htc_conv2d_2_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_htc_batch_normalization_2_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_htc_batch_normalization_2_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp9assignvariableop_24_htc_batch_normalization_2_moving_meanIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp=assignvariableop_25_htc_batch_normalization_2_moving_varianceIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_htc_conv2d_3_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_htc_conv2d_3_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_htc_batch_normalization_3_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_htc_batch_normalization_3_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp9assignvariableop_30_htc_batch_normalization_3_moving_meanIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp=assignvariableop_31_htc_batch_normalization_3_moving_varianceIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_htc_conv2d_4_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_htc_conv2d_4_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp3assignvariableop_34_htc_batch_normalization_4_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_htc_batch_normalization_4_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp9assignvariableop_36_htc_batch_normalization_4_moving_meanIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp=assignvariableop_37_htc_batch_normalization_4_moving_varianceIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_htc_dense_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp"assignvariableop_39_htc_dense_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp3assignvariableop_40_htc_batch_normalization_5_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp2assignvariableop_41_htc_batch_normalization_5_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp9assignvariableop_42_htc_batch_normalization_5_moving_meanIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp=assignvariableop_43_htc_batch_normalization_5_moving_varianceIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_iterationIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp!assignvariableop_45_learning_rateIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_m_htc_conv2d_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_v_htc_conv2d_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_htc_conv2d_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_htc_conv2d_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adam_m_htc_batch_normalization_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp8assignvariableop_51_adam_v_htc_batch_normalization_gammaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_m_htc_batch_normalization_betaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_v_htc_batch_normalization_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp.assignvariableop_54_adam_m_htc_conv2d_1_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp.assignvariableop_55_adam_v_htc_conv2d_1_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_m_htc_conv2d_1_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_v_htc_conv2d_1_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp:assignvariableop_58_adam_m_htc_batch_normalization_1_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp:assignvariableop_59_adam_v_htc_batch_normalization_1_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp9assignvariableop_60_adam_m_htc_batch_normalization_1_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp9assignvariableop_61_adam_v_htc_batch_normalization_1_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp.assignvariableop_62_adam_m_htc_conv2d_2_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp.assignvariableop_63_adam_v_htc_conv2d_2_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_m_htc_conv2d_2_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_v_htc_conv2d_2_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp:assignvariableop_66_adam_m_htc_batch_normalization_2_gammaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp:assignvariableop_67_adam_v_htc_batch_normalization_2_gammaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp9assignvariableop_68_adam_m_htc_batch_normalization_2_betaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp9assignvariableop_69_adam_v_htc_batch_normalization_2_betaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp.assignvariableop_70_adam_m_htc_conv2d_3_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_v_htc_conv2d_3_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp,assignvariableop_72_adam_m_htc_conv2d_3_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adam_v_htc_conv2d_3_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp:assignvariableop_74_adam_m_htc_batch_normalization_3_gammaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp:assignvariableop_75_adam_v_htc_batch_normalization_3_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp9assignvariableop_76_adam_m_htc_batch_normalization_3_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp9assignvariableop_77_adam_v_htc_batch_normalization_3_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp.assignvariableop_78_adam_m_htc_conv2d_4_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_v_htc_conv2d_4_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_m_htc_conv2d_4_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_v_htc_conv2d_4_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp:assignvariableop_82_adam_m_htc_batch_normalization_4_gammaIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp:assignvariableop_83_adam_v_htc_batch_normalization_4_gammaIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp9assignvariableop_84_adam_m_htc_batch_normalization_4_betaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp9assignvariableop_85_adam_v_htc_batch_normalization_4_betaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adam_m_htc_dense_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_v_htc_dense_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_m_htc_dense_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp)assignvariableop_89_adam_v_htc_dense_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp:assignvariableop_90_adam_m_htc_batch_normalization_5_gammaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp:assignvariableop_91_adam_v_htc_batch_normalization_5_gammaIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp9assignvariableop_92_adam_m_htc_batch_normalization_5_betaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp9assignvariableop_93_adam_v_htc_batch_normalization_5_betaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ?
Identity_94Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_95IdentityIdentity_94:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93*"
_acd_function_control_output(*
_output_shapes
 "#
identity_95Identity_95:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_93:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
I
-__inference_max_pooling2d_layer_call_fn_48128

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44333?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_48335

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_44988

inputs1
matmul_readvariableop_resource:	?$-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48325

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_48229

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44409?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_44890

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44465

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?$
__inference_test_step_45950

images

labelsC
)htc_conv2d_conv2d_readvariableop_resource: 8
*htc_conv2d_biasadd_readvariableop_resource: =
/htc_batch_normalization_readvariableop_resource: ?
1htc_batch_normalization_readvariableop_1_resource: N
@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource: P
Bhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: E
+htc_conv2d_1_conv2d_readvariableop_resource: @:
,htc_conv2d_1_biasadd_readvariableop_resource:@?
1htc_batch_normalization_1_readvariableop_resource:@A
3htc_batch_normalization_1_readvariableop_1_resource:@P
Bhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@R
Dhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@F
+htc_conv2d_2_conv2d_readvariableop_resource:@?;
,htc_conv2d_2_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_2_readvariableop_resource:	?B
3htc_batch_normalization_2_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	?G
+htc_conv2d_3_conv2d_readvariableop_resource:??;
,htc_conv2d_3_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_3_readvariableop_resource:	?B
3htc_batch_normalization_3_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?G
+htc_conv2d_4_conv2d_readvariableop_resource:??;
,htc_conv2d_4_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_4_readvariableop_resource:	?B
3htc_batch_normalization_4_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	?;
(htc_dense_matmul_readvariableop_resource:	?$7
)htc_dense_biasadd_readvariableop_resource:D
6htc_batch_normalization_5_cast_readvariableop_resource:F
8htc_batch_normalization_5_cast_1_readvariableop_resource:F
8htc_batch_normalization_5_cast_2_readvariableop_resource:F
8htc_batch_normalization_5_cast_3_readvariableop_resource:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: ??AssignAddVariableOp?AssignAddVariableOp_1?AssignAddVariableOp_2?AssignAddVariableOp_3?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?div_no_nan_1/ReadVariableOp?div_no_nan_1/ReadVariableOp_1?7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp?9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?&htc/batch_normalization/ReadVariableOp?(htc/batch_normalization/ReadVariableOp_1?9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_1/ReadVariableOp?*htc/batch_normalization_1/ReadVariableOp_1?9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_2/ReadVariableOp?*htc/batch_normalization_2/ReadVariableOp_1?9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_3/ReadVariableOp?*htc/batch_normalization_3/ReadVariableOp_1?9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_4/ReadVariableOp?*htc/batch_normalization_4/ReadVariableOp_1?-htc/batch_normalization_5/Cast/ReadVariableOp?/htc/batch_normalization_5/Cast_1/ReadVariableOp?/htc/batch_normalization_5/Cast_2/ReadVariableOp?/htc/batch_normalization_5/Cast_3/ReadVariableOp?!htc/conv2d/BiasAdd/ReadVariableOp? htc/conv2d/Conv2D/ReadVariableOp?#htc/conv2d_1/BiasAdd/ReadVariableOp?"htc/conv2d_1/Conv2D/ReadVariableOp?#htc/conv2d_2/BiasAdd/ReadVariableOp?"htc/conv2d_2/Conv2D/ReadVariableOp?#htc/conv2d_3/BiasAdd/ReadVariableOp?"htc/conv2d_3/Conv2D/ReadVariableOp?#htc/conv2d_4/BiasAdd/ReadVariableOp?"htc/conv2d_4/Conv2D/ReadVariableOp? htc/dense/BiasAdd/ReadVariableOp?htc/dense/MatMul/ReadVariableOpZ
htc/CastCastimages*

DstT0*

SrcT0*(
_output_shapes
: ???
 htc/conv2d/Conv2D/ReadVariableOpReadVariableOp)htc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
htc/conv2d/Conv2DConv2Dhtc/Cast:y:0(htc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
: ?? *
paddingVALID*
strides
?
!htc/conv2d/BiasAdd/ReadVariableOpReadVariableOp*htc_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
htc/conv2d/BiasAddBiasAddhtc/conv2d/Conv2D:output:0)htc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
: ?? ?
htc/max_pooling2d/MaxPoolMaxPoolhtc/conv2d/BiasAdd:output:0*&
_output_shapes
: bb *
ksize
*
paddingVALID*
strides
?
&htc/batch_normalization/ReadVariableOpReadVariableOp/htc_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
(htc/batch_normalization/ReadVariableOp_1ReadVariableOp1htc_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(htc/batch_normalization/FusedBatchNormV3FusedBatchNormV3"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:00htc/batch_normalization/ReadVariableOp_1:value:0?htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ahtc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: bb : : : : :*
epsilon%o?:*
is_training( u
htc/re_lu/ReluRelu,htc/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: bb ?
"htc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
htc/conv2d_1/Conv2DConv2Dhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@*
paddingVALID*
strides
?
#htc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
htc/conv2d_1/BiasAddBiasAddhtc/conv2d_1/Conv2D:output:0+htc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@?
htc/max_pooling2d_1/MaxPoolMaxPoolhtc/conv2d_1/BiasAdd:output:0*&
_output_shapes
: @*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_1/ReadVariableOpReadVariableOp1htc_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
*htc/batch_normalization_1/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
*htc/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:02htc/batch_normalization_1/ReadVariableOp_1:value:0Ahtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: @:@:@:@:@:*
epsilon%o?:*
is_training( y
htc/re_lu_1/ReluRelu.htc/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: @?
"htc/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
htc/conv2d_2/Conv2DConv2Dhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
#htc/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_2/BiasAddBiasAddhtc/conv2d_2/Conv2D:output:0+htc/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ??
htc/max_pooling2d_2/MaxPoolMaxPoolhtc/conv2d_2/BiasAdd:output:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_2/ReadVariableOpReadVariableOp1htc_batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_2/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:02htc/batch_normalization_2/ReadVariableOp_1:value:0Ahtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: ?:?:?:?:?:*
epsilon%o?:*
is_training( z
htc/re_lu_2/ReluRelu.htc/batch_normalization_2/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: ??
"htc/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
htc/conv2d_3/Conv2DConv2Dhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
#htc/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_3/BiasAddBiasAddhtc/conv2d_3/Conv2D:output:0+htc/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ??
htc/max_pooling2d_3/MaxPoolMaxPoolhtc/conv2d_3/BiasAdd:output:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_3/ReadVariableOpReadVariableOp1htc_batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_3/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:02htc/batch_normalization_3/ReadVariableOp_1:value:0Ahtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: ?:?:?:?:?:*
epsilon%o?:*
is_training( z
htc/re_lu_3/ReluRelu.htc/batch_normalization_3/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: ??
"htc/conv2d_4/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
htc/conv2d_4/Conv2DConv2Dhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
#htc/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_4/BiasAddBiasAddhtc/conv2d_4/Conv2D:output:0+htc/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ??
htc/max_pooling2d_4/MaxPoolMaxPoolhtc/conv2d_4/BiasAdd:output:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_4/ReadVariableOpReadVariableOp1htc_batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_4/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:02htc/batch_normalization_4/ReadVariableOp_1:value:0Ahtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: ?:?:?:?:?:*
epsilon%o?:*
is_training( z
htc/re_lu_4/ReluRelu.htc/batch_normalization_4/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: ?k
htc/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
htc/transpose	Transposehtc/re_lu_4/Relu:activations:0htc/transpose/perm:output:0*
T0*'
_output_shapes
: ?e
htc/dropout/IdentityIdentityhtc/transpose:y:0*
T0*'
_output_shapes
: ?m
htc/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
htc/transpose_1	Transposehtc/dropout/Identity:output:0htc/transpose_1/perm:output:0*
T0*'
_output_shapes
: ?b
htc/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   y
htc/flatten/ReshapeReshapehtc/transpose_1:y:0htc/flatten/Const:output:0*
T0*
_output_shapes
:	 ?$?
htc/dense/MatMul/ReadVariableOpReadVariableOp(htc_dense_matmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0?
htc/dense/MatMulMatMulhtc/flatten/Reshape:output:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ?
 htc/dense/BiasAdd/ReadVariableOpReadVariableOp)htc_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
htc/dense/BiasAddBiasAddhtc/dense/MatMul:product:0(htc/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: a
htc/dense/SoftmaxSoftmaxhtc/dense/BiasAdd:output:0*
T0*
_output_shapes

: ?
-htc/batch_normalization_5/Cast/ReadVariableOpReadVariableOp6htc_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0n
)htc/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'htc/batch_normalization_5/batchnorm/addAddV27htc/batch_normalization_5/Cast_1/ReadVariableOp:value:02htc/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/RsqrtRsqrt+htc/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:?
'htc/batch_normalization_5/batchnorm/mulMul-htc/batch_normalization_5/batchnorm/Rsqrt:y:07htc/batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/mul_1Mulhtc/dense/Softmax:softmax:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes

: ?
)htc/batch_normalization_5/batchnorm/mul_2Mul5htc/batch_normalization_5/Cast/ReadVariableOp:value:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:?
'htc/batch_normalization_5/batchnorm/subSub7htc/batch_normalization_5/Cast_2/ReadVariableOp:value:0-htc/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/add_1AddV2-htc/batch_normalization_5/batchnorm/mul_1:z:0+htc/batch_normalization_5/batchnorm/sub:z:0*
T0*
_output_shapes

: y
htc/activation/SoftmaxSoftmax-htc/batch_normalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes

: h
$sparse_categorical_crossentropy/CastCastlabels*

DstT0	*

SrcT0*
_output_shapes
: v
%sparse_categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits-htc/batch_normalization_5/batchnorm/add_1:z:0(sparse_categorical_crossentropy/Cast:y:0*
T0*$
_output_shapes
: : x
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: |
:sparse_categorical_crossentropy/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B : ?
?sparse_categorical_crossentropy/weighted_loss/num_elements/CastCastCsparse_categorical_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: t
2sparse_categorical_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: ?
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: ?
3sparse_categorical_crossentropy/weighted_loss/valueDivNoNan<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: ?
SumSum7sparse_categorical_crossentropy/weighted_loss/value:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: ?
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: J
Cast_1Castlabels*

DstT0*

SrcT0*
_output_shapes
: O
ShapeConst*
_output_shapes
:*
dtype0*
valueB: [
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????r
ArgMaxArgMax htc/activation/Softmax:softmax:0ArgMax/dimension:output:0*
T0*
_output_shapes
: S
Cast_2CastArgMax:output:0*

DstT0*

SrcT0	*
_output_shapes
: K
EqualEqual
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: M
Cast_3Cast	Equal:z:0*

DstT0*

SrcT0
*
_output_shapes
: O
ConstConst*
_output_shapes
:*
dtype0*
valueB: q
Sum_1Sum
Cast_3:y:0Const:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: ?
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_1:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0H
Size_1Const*
_output_shapes
: *
dtype0*
value	B : O
Cast_4CastSize_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_4:y:0^AssignAddVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0?
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2^AssignAddVariableOp_3*
_output_shapes
: *
dtype0?
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0?
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: *(
_construction_contextkEagerRuntime*}
_input_shapesl
j: ??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_326
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_12r
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp2v
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_19htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_12P
&htc/batch_normalization/ReadVariableOp&htc/batch_normalization/ReadVariableOp2T
(htc/batch_normalization/ReadVariableOp_1(htc/batch_normalization/ReadVariableOp_12v
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_1/ReadVariableOp(htc/batch_normalization_1/ReadVariableOp2X
*htc/batch_normalization_1/ReadVariableOp_1*htc/batch_normalization_1/ReadVariableOp_12v
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_2/ReadVariableOp(htc/batch_normalization_2/ReadVariableOp2X
*htc/batch_normalization_2/ReadVariableOp_1*htc/batch_normalization_2/ReadVariableOp_12v
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_3/ReadVariableOp(htc/batch_normalization_3/ReadVariableOp2X
*htc/batch_normalization_3/ReadVariableOp_1*htc/batch_normalization_3/ReadVariableOp_12v
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_4/ReadVariableOp(htc/batch_normalization_4/ReadVariableOp2X
*htc/batch_normalization_4/ReadVariableOp_1*htc/batch_normalization_4/ReadVariableOp_12^
-htc/batch_normalization_5/Cast/ReadVariableOp-htc/batch_normalization_5/Cast/ReadVariableOp2b
/htc/batch_normalization_5/Cast_1/ReadVariableOp/htc/batch_normalization_5/Cast_1/ReadVariableOp2b
/htc/batch_normalization_5/Cast_2/ReadVariableOp/htc/batch_normalization_5/Cast_2/ReadVariableOp2b
/htc/batch_normalization_5/Cast_3/ReadVariableOp/htc/batch_normalization_5/Cast_3/ReadVariableOp2F
!htc/conv2d/BiasAdd/ReadVariableOp!htc/conv2d/BiasAdd/ReadVariableOp2D
 htc/conv2d/Conv2D/ReadVariableOp htc/conv2d/Conv2D/ReadVariableOp2J
#htc/conv2d_1/BiasAdd/ReadVariableOp#htc/conv2d_1/BiasAdd/ReadVariableOp2H
"htc/conv2d_1/Conv2D/ReadVariableOp"htc/conv2d_1/Conv2D/ReadVariableOp2J
#htc/conv2d_2/BiasAdd/ReadVariableOp#htc/conv2d_2/BiasAdd/ReadVariableOp2H
"htc/conv2d_2/Conv2D/ReadVariableOp"htc/conv2d_2/Conv2D/ReadVariableOp2J
#htc/conv2d_3/BiasAdd/ReadVariableOp#htc/conv2d_3/BiasAdd/ReadVariableOp2H
"htc/conv2d_3/Conv2D/ReadVariableOp"htc/conv2d_3/Conv2D/ReadVariableOp2J
#htc/conv2d_4/BiasAdd/ReadVariableOp#htc/conv2d_4/BiasAdd/ReadVariableOp2H
"htc/conv2d_4/Conv2D/ReadVariableOp"htc/conv2d_4/Conv2D/ReadVariableOp2D
 htc/dense/BiasAdd/ReadVariableOp htc/dense/BiasAdd/ReadVariableOp2B
htc/dense/MatMul/ReadVariableOphtc/dense/MatMul/ReadVariableOp:P L
(
_output_shapes
: ??
 
_user_specified_nameimages:B>

_output_shapes
: 
 
_user_specified_namelabels
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48379

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_4_layer_call_fn_48550

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44662?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_4_layer_call_fn_48563

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44693?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_48306

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_48614

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44965i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
"__inference__update_step_xla_46816
gradient#
variable:@?!
readvariableop_resource:	 #
readvariableop_1_resource: 8
sub_2_readvariableop_resource:@?8
sub_3_readvariableop_resource:@???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: {
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*'
_output_shapes
:@?*
dtype0f
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=[
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*'
_output_shapes
:@??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0L
SquareSquaregradient*
T0*'
_output_shapes
:@?{
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*'
_output_shapes
:@?*
dtype0h
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:[
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*'
_output_shapes
:@??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*'
_output_shapes
:@?*
dtype0e
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*'
_output_shapes
:@??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*'
_output_shapes
:@?*
dtype0_
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3^
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:@?\
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*'
_output_shapes
:@?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*0
_input_shapes
:@?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:Q M
'
_output_shapes
:@?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
C
'__inference_flatten_layer_call_fn_48641

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44975a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?!
 __inference__wrapped_model_44324
input_1C
)htc_conv2d_conv2d_readvariableop_resource: 8
*htc_conv2d_biasadd_readvariableop_resource: =
/htc_batch_normalization_readvariableop_resource: ?
1htc_batch_normalization_readvariableop_1_resource: N
@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource: P
Bhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: E
+htc_conv2d_1_conv2d_readvariableop_resource: @:
,htc_conv2d_1_biasadd_readvariableop_resource:@?
1htc_batch_normalization_1_readvariableop_resource:@A
3htc_batch_normalization_1_readvariableop_1_resource:@P
Bhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@R
Dhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@F
+htc_conv2d_2_conv2d_readvariableop_resource:@?;
,htc_conv2d_2_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_2_readvariableop_resource:	?B
3htc_batch_normalization_2_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	?G
+htc_conv2d_3_conv2d_readvariableop_resource:??;
,htc_conv2d_3_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_3_readvariableop_resource:	?B
3htc_batch_normalization_3_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?G
+htc_conv2d_4_conv2d_readvariableop_resource:??;
,htc_conv2d_4_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_4_readvariableop_resource:	?B
3htc_batch_normalization_4_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	?;
(htc_dense_matmul_readvariableop_resource:	?$7
)htc_dense_biasadd_readvariableop_resource:D
6htc_batch_normalization_5_cast_readvariableop_resource:F
8htc_batch_normalization_5_cast_1_readvariableop_resource:F
8htc_batch_normalization_5_cast_2_readvariableop_resource:F
8htc_batch_normalization_5_cast_3_readvariableop_resource:
identity??7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp?9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?&htc/batch_normalization/ReadVariableOp?(htc/batch_normalization/ReadVariableOp_1?9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_1/ReadVariableOp?*htc/batch_normalization_1/ReadVariableOp_1?9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_2/ReadVariableOp?*htc/batch_normalization_2/ReadVariableOp_1?9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_3/ReadVariableOp?*htc/batch_normalization_3/ReadVariableOp_1?9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_4/ReadVariableOp?*htc/batch_normalization_4/ReadVariableOp_1?-htc/batch_normalization_5/Cast/ReadVariableOp?/htc/batch_normalization_5/Cast_1/ReadVariableOp?/htc/batch_normalization_5/Cast_2/ReadVariableOp?/htc/batch_normalization_5/Cast_3/ReadVariableOp?!htc/conv2d/BiasAdd/ReadVariableOp? htc/conv2d/Conv2D/ReadVariableOp?#htc/conv2d_1/BiasAdd/ReadVariableOp?"htc/conv2d_1/Conv2D/ReadVariableOp?#htc/conv2d_2/BiasAdd/ReadVariableOp?"htc/conv2d_2/Conv2D/ReadVariableOp?#htc/conv2d_3/BiasAdd/ReadVariableOp?"htc/conv2d_3/Conv2D/ReadVariableOp?#htc/conv2d_4/BiasAdd/ReadVariableOp?"htc/conv2d_4/Conv2D/ReadVariableOp? htc/dense/BiasAdd/ReadVariableOp?htc/dense/MatMul/ReadVariableOp?
 htc/conv2d/Conv2D/ReadVariableOpReadVariableOp)htc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
htc/conv2d/Conv2DConv2Dinput_1(htc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
?
!htc/conv2d/BiasAdd/ReadVariableOpReadVariableOp*htc_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
htc/conv2d/BiasAddBiasAddhtc/conv2d/Conv2D:output:0)htc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
htc/max_pooling2d/MaxPoolMaxPoolhtc/conv2d/BiasAdd:output:0*/
_output_shapes
:?????????bb *
ksize
*
paddingVALID*
strides
?
&htc/batch_normalization/ReadVariableOpReadVariableOp/htc_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
(htc/batch_normalization/ReadVariableOp_1ReadVariableOp1htc_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(htc/batch_normalization/FusedBatchNormV3FusedBatchNormV3"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:00htc/batch_normalization/ReadVariableOp_1:value:0?htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ahtc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????bb : : : : :*
epsilon%o?:*
is_training( ~
htc/re_lu/ReluRelu,htc/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????bb ?
"htc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
htc/conv2d_1/Conv2DConv2Dhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@*
paddingVALID*
strides
?
#htc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
htc/conv2d_1/BiasAddBiasAddhtc/conv2d_1/Conv2D:output:0+htc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@?
htc/max_pooling2d_1/MaxPoolMaxPoolhtc/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_1/ReadVariableOpReadVariableOp1htc_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
*htc/batch_normalization_1/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
*htc/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:02htc/batch_normalization_1/ReadVariableOp_1:value:0Ahtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
htc/re_lu_1/ReluRelu.htc/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@?
"htc/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
htc/conv2d_2/Conv2DConv2Dhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
#htc/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_2/BiasAddBiasAddhtc/conv2d_2/Conv2D:output:0+htc/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
htc/max_pooling2d_2/MaxPoolMaxPoolhtc/conv2d_2/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_2/ReadVariableOpReadVariableOp1htc_batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_2/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:02htc/batch_normalization_2/ReadVariableOp_1:value:0Ahtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
htc/re_lu_2/ReluRelu.htc/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:???????????
"htc/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
htc/conv2d_3/Conv2DConv2Dhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
#htc/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_3/BiasAddBiasAddhtc/conv2d_3/Conv2D:output:0+htc/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
htc/max_pooling2d_3/MaxPoolMaxPoolhtc/conv2d_3/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_3/ReadVariableOpReadVariableOp1htc_batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_3/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:02htc/batch_normalization_3/ReadVariableOp_1:value:0Ahtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
htc/re_lu_3/ReluRelu.htc/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:???????????
"htc/conv2d_4/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
htc/conv2d_4/Conv2DConv2Dhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
#htc/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_4/BiasAddBiasAddhtc/conv2d_4/Conv2D:output:0+htc/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
htc/max_pooling2d_4/MaxPoolMaxPoolhtc/conv2d_4/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_4/ReadVariableOpReadVariableOp1htc_batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_4/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:02htc/batch_normalization_4/ReadVariableOp_1:value:0Ahtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
htc/re_lu_4/ReluRelu.htc/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????k
htc/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
htc/transpose	Transposehtc/re_lu_4/Relu:activations:0htc/transpose/perm:output:0*
T0*0
_output_shapes
:??????????n
htc/dropout/IdentityIdentityhtc/transpose:y:0*
T0*0
_output_shapes
:??????????m
htc/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
htc/transpose_1	Transposehtc/dropout/Identity:output:0htc/transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????b
htc/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
htc/flatten/ReshapeReshapehtc/transpose_1:y:0htc/flatten/Const:output:0*
T0*(
_output_shapes
:??????????$?
htc/dense/MatMul/ReadVariableOpReadVariableOp(htc_dense_matmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0?
htc/dense/MatMulMatMulhtc/flatten/Reshape:output:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 htc/dense/BiasAdd/ReadVariableOpReadVariableOp)htc_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
htc/dense/BiasAddBiasAddhtc/dense/MatMul:product:0(htc/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
htc/dense/SoftmaxSoftmaxhtc/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
-htc/batch_normalization_5/Cast/ReadVariableOpReadVariableOp6htc_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0n
)htc/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'htc/batch_normalization_5/batchnorm/addAddV27htc/batch_normalization_5/Cast_1/ReadVariableOp:value:02htc/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/RsqrtRsqrt+htc/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:?
'htc/batch_normalization_5/batchnorm/mulMul-htc/batch_normalization_5/batchnorm/Rsqrt:y:07htc/batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/mul_1Mulhtc/dense/Softmax:softmax:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
)htc/batch_normalization_5/batchnorm/mul_2Mul5htc/batch_normalization_5/Cast/ReadVariableOp:value:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:?
'htc/batch_normalization_5/batchnorm/subSub7htc/batch_normalization_5/Cast_2/ReadVariableOp:value:0-htc/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/add_1AddV2-htc/batch_normalization_5/batchnorm/mul_1:z:0+htc/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
htc/activation/SoftmaxSoftmax-htc/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity htc/activation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp8^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1'^htc/batch_normalization/ReadVariableOp)^htc/batch_normalization/ReadVariableOp_1:^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_1/ReadVariableOp+^htc/batch_normalization_1/ReadVariableOp_1:^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_2/ReadVariableOp+^htc/batch_normalization_2/ReadVariableOp_1:^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_3/ReadVariableOp+^htc/batch_normalization_3/ReadVariableOp_1:^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_4/ReadVariableOp+^htc/batch_normalization_4/ReadVariableOp_1.^htc/batch_normalization_5/Cast/ReadVariableOp0^htc/batch_normalization_5/Cast_1/ReadVariableOp0^htc/batch_normalization_5/Cast_2/ReadVariableOp0^htc/batch_normalization_5/Cast_3/ReadVariableOp"^htc/conv2d/BiasAdd/ReadVariableOp!^htc/conv2d/Conv2D/ReadVariableOp$^htc/conv2d_1/BiasAdd/ReadVariableOp#^htc/conv2d_1/Conv2D/ReadVariableOp$^htc/conv2d_2/BiasAdd/ReadVariableOp#^htc/conv2d_2/Conv2D/ReadVariableOp$^htc/conv2d_3/BiasAdd/ReadVariableOp#^htc/conv2d_3/Conv2D/ReadVariableOp$^htc/conv2d_4/BiasAdd/ReadVariableOp#^htc/conv2d_4/Conv2D/ReadVariableOp!^htc/dense/BiasAdd/ReadVariableOp ^htc/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp2v
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_19htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_12P
&htc/batch_normalization/ReadVariableOp&htc/batch_normalization/ReadVariableOp2T
(htc/batch_normalization/ReadVariableOp_1(htc/batch_normalization/ReadVariableOp_12v
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_1/ReadVariableOp(htc/batch_normalization_1/ReadVariableOp2X
*htc/batch_normalization_1/ReadVariableOp_1*htc/batch_normalization_1/ReadVariableOp_12v
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_2/ReadVariableOp(htc/batch_normalization_2/ReadVariableOp2X
*htc/batch_normalization_2/ReadVariableOp_1*htc/batch_normalization_2/ReadVariableOp_12v
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_3/ReadVariableOp(htc/batch_normalization_3/ReadVariableOp2X
*htc/batch_normalization_3/ReadVariableOp_1*htc/batch_normalization_3/ReadVariableOp_12v
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_4/ReadVariableOp(htc/batch_normalization_4/ReadVariableOp2X
*htc/batch_normalization_4/ReadVariableOp_1*htc/batch_normalization_4/ReadVariableOp_12^
-htc/batch_normalization_5/Cast/ReadVariableOp-htc/batch_normalization_5/Cast/ReadVariableOp2b
/htc/batch_normalization_5/Cast_1/ReadVariableOp/htc/batch_normalization_5/Cast_1/ReadVariableOp2b
/htc/batch_normalization_5/Cast_2/ReadVariableOp/htc/batch_normalization_5/Cast_2/ReadVariableOp2b
/htc/batch_normalization_5/Cast_3/ReadVariableOp/htc/batch_normalization_5/Cast_3/ReadVariableOp2F
!htc/conv2d/BiasAdd/ReadVariableOp!htc/conv2d/BiasAdd/ReadVariableOp2D
 htc/conv2d/Conv2D/ReadVariableOp htc/conv2d/Conv2D/ReadVariableOp2J
#htc/conv2d_1/BiasAdd/ReadVariableOp#htc/conv2d_1/BiasAdd/ReadVariableOp2H
"htc/conv2d_1/Conv2D/ReadVariableOp"htc/conv2d_1/Conv2D/ReadVariableOp2J
#htc/conv2d_2/BiasAdd/ReadVariableOp#htc/conv2d_2/BiasAdd/ReadVariableOp2H
"htc/conv2d_2/Conv2D/ReadVariableOp"htc/conv2d_2/Conv2D/ReadVariableOp2J
#htc/conv2d_3/BiasAdd/ReadVariableOp#htc/conv2d_3/BiasAdd/ReadVariableOp2H
"htc/conv2d_3/Conv2D/ReadVariableOp"htc/conv2d_3/Conv2D/ReadVariableOp2J
#htc/conv2d_4/BiasAdd/ReadVariableOp#htc/conv2d_4/BiasAdd/ReadVariableOp2H
"htc/conv2d_4/Conv2D/ReadVariableOp"htc/conv2d_4/Conv2D/ReadVariableOp2D
 htc/dense/BiasAdd/ReadVariableOp htc/dense/BiasAdd/ReadVariableOp2B
htc/dense/MatMul/ReadVariableOphtc/dense/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_48636

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48581

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_48508

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44662

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?s
?
>__inference_htc_layer_call_and_return_conditional_losses_45011
x&
conv2d_44804: 
conv2d_44806: '
batch_normalization_44810: '
batch_normalization_44812: '
batch_normalization_44814: '
batch_normalization_44816: (
conv2d_1_44837: @
conv2d_1_44839:@)
batch_normalization_1_44843:@)
batch_normalization_1_44845:@)
batch_normalization_1_44847:@)
batch_normalization_1_44849:@)
conv2d_2_44870:@?
conv2d_2_44872:	?*
batch_normalization_2_44876:	?*
batch_normalization_2_44878:	?*
batch_normalization_2_44880:	?*
batch_normalization_2_44882:	?*
conv2d_3_44903:??
conv2d_3_44905:	?*
batch_normalization_3_44909:	?*
batch_normalization_3_44911:	?*
batch_normalization_3_44913:	?*
batch_normalization_3_44915:	?*
conv2d_4_44936:??
conv2d_4_44938:	?*
batch_normalization_4_44942:	?*
batch_normalization_4_44944:	?*
batch_normalization_4_44946:	?*
batch_normalization_4_44948:	?
dense_44989:	?$
dense_44991:)
batch_normalization_5_44994:)
batch_normalization_5_44996:)
batch_normalization_5_44998:)
batch_normalization_5_45000:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_44804conv2d_44806*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44803?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44333?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_44810batch_normalization_44812batch_normalization_44814batch_normalization_44816*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44358?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_44824?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_44837conv2d_1_44839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44836?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44409?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_44843batch_normalization_1_44845batch_normalization_1_44847batch_normalization_1_44849*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44434?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_44857?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_44870conv2d_2_44872*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44869?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44485?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_44876batch_normalization_2_44878batch_normalization_2_44880batch_normalization_2_44882*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44510?
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_44890?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_44903conv2d_3_44905*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44902?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44561?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_44909batch_normalization_3_44911batch_normalization_3_44913batch_normalization_3_44915*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44586?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_44923?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_44936conv2d_4_44938*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44935?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_44637?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_44942batch_normalization_4_44944batch_normalization_4_44946batch_normalization_4_44948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44662?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_44956g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:???????????
dropout/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44965i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_1	Transpose dropout/PartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:???????????
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44975?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_44989dense_44991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44988?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_44994batch_normalization_5_44996batch_normalization_5_44998batch_normalization_5_45000*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44728?
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_45008r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
#__inference_signature_wrapper_47639
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?

unknown_29:	?$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *)
f$R"
 __inference__wrapped_model_44324o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
K
/__inference_max_pooling2d_2_layer_call_fn_48330

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44485?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_46675
gradient
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:@+
sub_3_readvariableop_resource:@??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:@*
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:@?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
:@n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:@*
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:@?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:@*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:@?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:@*
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:@O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:@f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:@: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
??
?#
>__inference_htc_layer_call_and_return_conditional_losses_48104
x?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_2_conv2d_readvariableop_resource:@?7
(conv2d_2_biasadd_readvariableop_resource:	?<
-batch_normalization_2_readvariableop_resource:	?>
/batch_normalization_2_readvariableop_1_resource:	?M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_3_conv2d_readvariableop_resource:??7
(conv2d_3_biasadd_readvariableop_resource:	?<
-batch_normalization_3_readvariableop_resource:	?>
/batch_normalization_3_readvariableop_1_resource:	?M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_4_conv2d_readvariableop_resource:??7
(conv2d_4_biasadd_readvariableop_resource:	?<
-batch_normalization_4_readvariableop_resource:	?>
/batch_normalization_4_readvariableop_1_resource:	?M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	?7
$dense_matmul_readvariableop_resource:	?$3
%dense_biasadd_readvariableop_resource:K
=batch_normalization_5_assignmovingavg_readvariableop_resource:M
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_5_cast_readvariableop_resource:B
4batch_normalization_5_cast_1_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?%batch_normalization_5/AssignMovingAvg?4batch_normalization_5/AssignMovingAvg/ReadVariableOp?'batch_normalization_5/AssignMovingAvg_1?6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_5/Cast/ReadVariableOp?+batch_normalization_5/Cast_1/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:?????????bb *
ksize
*
paddingVALID*
strides
?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????bb : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????bb ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:???????????
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:???????????
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_4/Conv2DConv2Dre_lu_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/BiasAdd:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transposere_lu_4/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMultranspose:y:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????R
dropout/dropout/ShapeShapetranspose:y:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*0
_output_shapes
:??????????i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_1	Transpose!dropout/dropout/SelectV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   v
flatten/ReshapeReshapetranspose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????$?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_5/moments/meanMeandense/Softmax:softmax:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:?
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense/Softmax:softmax:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0?
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:?
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_5/batchnorm/mul_1Muldense/Softmax:softmax:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:?
#batch_normalization_5/batchnorm/subSub1batch_normalization_5/Cast/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????z
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????k
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_5/Cast/ReadVariableOp)batch_normalization_5/Cast/ReadVariableOp2Z
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48397

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_47474
gradient
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:+
sub_3_readvariableop_resource:??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:*
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
:n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:*
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:*
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_48133

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_44803

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
\
@__inference_re_lu_layer_call_and_return_conditional_losses_48205

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????bb b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????bb "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????bb :W S
/
_output_shapes
:?????????bb 
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44935

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_48123

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44358

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44485

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_htc_layer_call_fn_47793
x!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?

unknown_29:	?$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
 #$*2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_htc_layer_call_and_return_conditional_losses_45395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
? 
?
"__inference__update_step_xla_46534
gradient
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource: +
sub_3_readvariableop_resource: ??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
: *
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
: n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
: ?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
: L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
: O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
: f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
\
@__inference_re_lu_layer_call_and_return_conditional_losses_44824

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????bb b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????bb "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????bb :W S
/
_output_shapes
:?????????bb 
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_45008

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44561

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?$
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44775

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44869

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_48146

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44358?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_3_layer_call_fn_48431

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44561?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_46722
gradient
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:@+
sub_3_readvariableop_resource:@??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:@*
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:@?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
:@n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:@*
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:@?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:@*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:@?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:@*
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:@O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:@f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:@: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?$
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48747

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_47286
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
(__inference_conv2d_1_layer_call_fn_48214

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44836w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????^^@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????bb : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????bb 
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_46910
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48224

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????^^@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????bb : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????bb 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44617

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_48537

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_47521
gradient
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:+
sub_3_readvariableop_resource:??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:*
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
:n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:*
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:*
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
(__inference_conv2d_3_layer_call_fn_48416

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44902x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_44956

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?t
?
>__inference_htc_layer_call_and_return_conditional_losses_45395
x&
conv2d_45293: 
conv2d_45295: '
batch_normalization_45299: '
batch_normalization_45301: '
batch_normalization_45303: '
batch_normalization_45305: (
conv2d_1_45309: @
conv2d_1_45311:@)
batch_normalization_1_45315:@)
batch_normalization_1_45317:@)
batch_normalization_1_45319:@)
batch_normalization_1_45321:@)
conv2d_2_45325:@?
conv2d_2_45327:	?*
batch_normalization_2_45331:	?*
batch_normalization_2_45333:	?*
batch_normalization_2_45335:	?*
batch_normalization_2_45337:	?*
conv2d_3_45341:??
conv2d_3_45343:	?*
batch_normalization_3_45347:	?*
batch_normalization_3_45349:	?*
batch_normalization_3_45351:	?*
batch_normalization_3_45353:	?*
conv2d_4_45357:??
conv2d_4_45359:	?*
batch_normalization_4_45363:	?*
batch_normalization_4_45365:	?*
batch_normalization_4_45367:	?*
batch_normalization_4_45369:	?
dense_45379:	?$
dense_45381:)
batch_normalization_5_45384:)
batch_normalization_5_45386:)
batch_normalization_5_45388:)
batch_normalization_5_45390:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_45293conv2d_45295*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44803?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44333?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_45299batch_normalization_45301batch_normalization_45303batch_normalization_45305*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44389?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_44824?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_45309conv2d_1_45311*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44836?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44409?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_45315batch_normalization_1_45317batch_normalization_1_45319batch_normalization_1_45321*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44465?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_44857?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_45325conv2d_2_45327*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44869?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44485?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_45331batch_normalization_2_45333batch_normalization_2_45335batch_normalization_2_45337*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44541?
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_44890?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_45341conv2d_3_45343*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44902?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44561?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_45347batch_normalization_3_45349batch_normalization_3_45351batch_normalization_3_45353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44617?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_44923?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_45357conv2d_4_45359*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44935?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_44637?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_45363batch_normalization_4_45365batch_normalization_4_45367batch_normalization_4_45369*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44693?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_44956g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:???????????
dropout/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45128i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_1	Transpose(dropout/StatefulPartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:???????????
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44975?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_45379dense_45381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44988?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_45384batch_normalization_5_45386batch_normalization_5_45388batch_normalization_5_45390*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44775?
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_45008r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48599

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_htc_layer_call_fn_47716
x!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?

unknown_29:	?$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_htc_layer_call_and_return_conditional_losses_45011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
(__inference_conv2d_2_layer_call_fn_48315

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44869x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?9
__inference_train_step_47560

images

labelsC
)htc_conv2d_conv2d_readvariableop_resource: 8
*htc_conv2d_biasadd_readvariableop_resource: =
/htc_batch_normalization_readvariableop_resource: ?
1htc_batch_normalization_readvariableop_1_resource: N
@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource: P
Bhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: E
+htc_conv2d_1_conv2d_readvariableop_resource: @:
,htc_conv2d_1_biasadd_readvariableop_resource:@?
1htc_batch_normalization_1_readvariableop_resource:@A
3htc_batch_normalization_1_readvariableop_1_resource:@P
Bhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@R
Dhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@F
+htc_conv2d_2_conv2d_readvariableop_resource:@?;
,htc_conv2d_2_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_2_readvariableop_resource:	?B
3htc_batch_normalization_2_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	?G
+htc_conv2d_3_conv2d_readvariableop_resource:??;
,htc_conv2d_3_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_3_readvariableop_resource:	?B
3htc_batch_normalization_3_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?G
+htc_conv2d_4_conv2d_readvariableop_resource:??;
,htc_conv2d_4_biasadd_readvariableop_resource:	?@
1htc_batch_normalization_4_readvariableop_resource:	?B
3htc_batch_normalization_4_readvariableop_1_resource:	?Q
Bhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	?S
Dhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	?;
(htc_dense_matmul_readvariableop_resource:	?$7
)htc_dense_biasadd_readvariableop_resource:O
Ahtc_batch_normalization_5_assignmovingavg_readvariableop_resource:Q
Chtc_batch_normalization_5_assignmovingavg_1_readvariableop_resource:D
6htc_batch_normalization_5_cast_readvariableop_resource:F
8htc_batch_normalization_5_cast_1_readvariableop_resource:
unknown:	 
	unknown_0: #
	unknown_1: #
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: @$

unknown_10: @

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?%

unknown_18:@?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?

unknown_24:	?&

unknown_25:??&

unknown_26:??

unknown_27:	?

unknown_28:	?

unknown_29:	?

unknown_30:	?

unknown_31:	?

unknown_32:	?&

unknown_33:??&

unknown_34:??

unknown_35:	?

unknown_36:	?

unknown_37:	?

unknown_38:	?

unknown_39:	?

unknown_40:	?

unknown_41:	?$

unknown_42:	?$

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:(
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: (
assignaddvariableop_4_resource: ??AssignAddVariableOp?AssignAddVariableOp_1?AssignAddVariableOp_2?AssignAddVariableOp_3?AssignAddVariableOp_4?StatefulPartitionedCall?StatefulPartitionedCall_1?StatefulPartitionedCall_10?StatefulPartitionedCall_11?StatefulPartitionedCall_12?StatefulPartitionedCall_13?StatefulPartitionedCall_14?StatefulPartitionedCall_15?StatefulPartitionedCall_16?StatefulPartitionedCall_17?StatefulPartitionedCall_18?StatefulPartitionedCall_19?StatefulPartitionedCall_2?StatefulPartitionedCall_20?StatefulPartitionedCall_21?StatefulPartitionedCall_22?StatefulPartitionedCall_23?StatefulPartitionedCall_3?StatefulPartitionedCall_4?StatefulPartitionedCall_5?StatefulPartitionedCall_6?StatefulPartitionedCall_7?StatefulPartitionedCall_8?StatefulPartitionedCall_9?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?div_no_nan_1/ReadVariableOp?div_no_nan_1/ReadVariableOp_1?&htc/batch_normalization/AssignNewValue?(htc/batch_normalization/AssignNewValue_1?7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp?9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?&htc/batch_normalization/ReadVariableOp?(htc/batch_normalization/ReadVariableOp_1?(htc/batch_normalization_1/AssignNewValue?*htc/batch_normalization_1/AssignNewValue_1?9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_1/ReadVariableOp?*htc/batch_normalization_1/ReadVariableOp_1?(htc/batch_normalization_2/AssignNewValue?*htc/batch_normalization_2/AssignNewValue_1?9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_2/ReadVariableOp?*htc/batch_normalization_2/ReadVariableOp_1?(htc/batch_normalization_3/AssignNewValue?*htc/batch_normalization_3/AssignNewValue_1?9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_3/ReadVariableOp?*htc/batch_normalization_3/ReadVariableOp_1?(htc/batch_normalization_4/AssignNewValue?*htc/batch_normalization_4/AssignNewValue_1?9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?(htc/batch_normalization_4/ReadVariableOp?*htc/batch_normalization_4/ReadVariableOp_1?)htc/batch_normalization_5/AssignMovingAvg?8htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp?+htc/batch_normalization_5/AssignMovingAvg_1?:htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp?-htc/batch_normalization_5/Cast/ReadVariableOp?/htc/batch_normalization_5/Cast_1/ReadVariableOp?!htc/conv2d/BiasAdd/ReadVariableOp? htc/conv2d/Conv2D/ReadVariableOp?#htc/conv2d_1/BiasAdd/ReadVariableOp?"htc/conv2d_1/Conv2D/ReadVariableOp?#htc/conv2d_2/BiasAdd/ReadVariableOp?"htc/conv2d_2/Conv2D/ReadVariableOp?#htc/conv2d_3/BiasAdd/ReadVariableOp?"htc/conv2d_3/Conv2D/ReadVariableOp?#htc/conv2d_4/BiasAdd/ReadVariableOp?"htc/conv2d_4/Conv2D/ReadVariableOp? htc/dense/BiasAdd/ReadVariableOp?htc/dense/MatMul/ReadVariableOpZ
htc/CastCastimages*

DstT0*

SrcT0*(
_output_shapes
: ???
 htc/conv2d/Conv2D/ReadVariableOpReadVariableOp)htc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
htc/conv2d/Conv2DConv2Dhtc/Cast:y:0(htc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
: ?? *
paddingVALID*
strides
?
!htc/conv2d/BiasAdd/ReadVariableOpReadVariableOp*htc_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
htc/conv2d/BiasAddBiasAddhtc/conv2d/Conv2D:output:0)htc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
: ?? ?
htc/max_pooling2d/MaxPoolMaxPoolhtc/conv2d/BiasAdd:output:0*&
_output_shapes
: bb *
ksize
*
paddingVALID*
strides
?
&htc/batch_normalization/ReadVariableOpReadVariableOp/htc_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
(htc/batch_normalization/ReadVariableOp_1ReadVariableOp1htc_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(htc/batch_normalization/FusedBatchNormV3FusedBatchNormV3"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:00htc/batch_normalization/ReadVariableOp_1:value:0?htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ahtc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: bb : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
&htc/batch_normalization/AssignNewValueAssignVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource5htc/batch_normalization/FusedBatchNormV3:batch_mean:08^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
(htc/batch_normalization/AssignNewValue_1AssignVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource9htc/batch_normalization/FusedBatchNormV3:batch_variance:0:^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(u
htc/re_lu/ReluRelu,htc/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: bb ?
"htc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
htc/conv2d_1/Conv2DConv2Dhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@*
paddingVALID*
strides
?
#htc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
htc/conv2d_1/BiasAddBiasAddhtc/conv2d_1/Conv2D:output:0+htc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@?
htc/max_pooling2d_1/MaxPoolMaxPoolhtc/conv2d_1/BiasAdd:output:0*&
_output_shapes
: @*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_1/ReadVariableOpReadVariableOp1htc_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
*htc/batch_normalization_1/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
*htc/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:02htc/batch_normalization_1/ReadVariableOp_1:value:0Ahtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
(htc/batch_normalization_1/AssignNewValueAssignVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_1/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
*htc/batch_normalization_1/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_1/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(y
htc/re_lu_1/ReluRelu.htc/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: @?
"htc/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
htc/conv2d_2/Conv2DConv2Dhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
#htc/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_2/BiasAddBiasAddhtc/conv2d_2/Conv2D:output:0+htc/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ??
htc/max_pooling2d_2/MaxPoolMaxPoolhtc/conv2d_2/BiasAdd:output:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_2/ReadVariableOpReadVariableOp1htc_batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_2/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:02htc/batch_normalization_2/ReadVariableOp_1:value:0Ahtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
(htc/batch_normalization_2/AssignNewValueAssignVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_2/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
*htc/batch_normalization_2/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_2/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
htc/re_lu_2/ReluRelu.htc/batch_normalization_2/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: ??
"htc/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
htc/conv2d_3/Conv2DConv2Dhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
#htc/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_3/BiasAddBiasAddhtc/conv2d_3/Conv2D:output:0+htc/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ??
htc/max_pooling2d_3/MaxPoolMaxPoolhtc/conv2d_3/BiasAdd:output:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_3/ReadVariableOpReadVariableOp1htc_batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_3/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:02htc/batch_normalization_3/ReadVariableOp_1:value:0Ahtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
(htc/batch_normalization_3/AssignNewValueAssignVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_3/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
*htc/batch_normalization_3/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_3/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
htc/re_lu_3/ReluRelu.htc/batch_normalization_3/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: ??
"htc/conv2d_4/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
htc/conv2d_4/Conv2DConv2Dhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
#htc/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
htc/conv2d_4/BiasAddBiasAddhtc/conv2d_4/Conv2D:output:0+htc/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ??
htc/max_pooling2d_4/MaxPoolMaxPoolhtc/conv2d_4/BiasAdd:output:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
(htc/batch_normalization_4/ReadVariableOpReadVariableOp1htc_batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_4/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
*htc/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:02htc/batch_normalization_4/ReadVariableOp_1:value:0Ahtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
(htc/batch_normalization_4/AssignNewValueAssignVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_4/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
*htc/batch_normalization_4/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_4/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
htc/re_lu_4/ReluRelu.htc/batch_normalization_4/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: ?k
htc/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
htc/transpose	Transposehtc/re_lu_4/Relu:activations:0htc/transpose/perm:output:0*
T0*'
_output_shapes
: ?^
htc/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
htc/dropout/dropout/MulMulhtc/transpose:y:0"htc/dropout/dropout/Const:output:0*
T0*'
_output_shapes
: ?r
htc/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
0htc/dropout/dropout/random_uniform/RandomUniformRandomUniform"htc/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
: ?*
dtype0g
"htc/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 htc/dropout/dropout/GreaterEqualGreaterEqual9htc/dropout/dropout/random_uniform/RandomUniform:output:0+htc/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
: ?`
htc/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
htc/dropout/dropout/SelectV2SelectV2$htc/dropout/dropout/GreaterEqual:z:0htc/dropout/dropout/Mul:z:0$htc/dropout/dropout/Const_1:output:0*
T0*'
_output_shapes
: ?m
htc/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
htc/transpose_1	Transpose%htc/dropout/dropout/SelectV2:output:0htc/transpose_1/perm:output:0*
T0*'
_output_shapes
: ?b
htc/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   y
htc/flatten/ReshapeReshapehtc/transpose_1:y:0htc/flatten/Const:output:0*
T0*
_output_shapes
:	 ?$?
htc/dense/MatMul/ReadVariableOpReadVariableOp(htc_dense_matmul_readvariableop_resource*
_output_shapes
:	?$*
dtype0?
htc/dense/MatMulMatMulhtc/flatten/Reshape:output:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ?
 htc/dense/BiasAdd/ReadVariableOpReadVariableOp)htc_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
htc/dense/BiasAddBiasAddhtc/dense/MatMul:product:0(htc/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: a
htc/dense/SoftmaxSoftmaxhtc/dense/BiasAdd:output:0*
T0*
_output_shapes

: ?
8htc/batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&htc/batch_normalization_5/moments/meanMeanhtc/dense/Softmax:softmax:0Ahtc/batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
.htc/batch_normalization_5/moments/StopGradientStopGradient/htc/batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:?
3htc/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencehtc/dense/Softmax:softmax:07htc/batch_normalization_5/moments/StopGradient:output:0*
T0*
_output_shapes

: ?
<htc/batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
*htc/batch_normalization_5/moments/varianceMean7htc/batch_normalization_5/moments/SquaredDifference:z:0Ehtc/batch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
)htc/batch_normalization_5/moments/SqueezeSqueeze/htc/batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
+htc/batch_normalization_5/moments/Squeeze_1Squeeze3htc/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
/htc/batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
8htc/batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOpAhtc_batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
-htc/batch_normalization_5/AssignMovingAvg/subSub@htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:02htc/batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:?
-htc/batch_normalization_5/AssignMovingAvg/mulMul1htc/batch_normalization_5/AssignMovingAvg/sub:z:08htc/batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/AssignMovingAvgAssignSubVariableOpAhtc_batch_normalization_5_assignmovingavg_readvariableop_resource1htc/batch_normalization_5/AssignMovingAvg/mul:z:09^htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0v
1htc/batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
:htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOpChtc_batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/AssignMovingAvg_1/subSubBhtc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:04htc/batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
/htc/batch_normalization_5/AssignMovingAvg_1/mulMul3htc/batch_normalization_5/AssignMovingAvg_1/sub:z:0:htc/batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
+htc/batch_normalization_5/AssignMovingAvg_1AssignSubVariableOpChtc_batch_normalization_5_assignmovingavg_1_readvariableop_resource3htc/batch_normalization_5/AssignMovingAvg_1/mul:z:0;^htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0?
-htc/batch_normalization_5/Cast/ReadVariableOpReadVariableOp6htc_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0?
/htc/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0n
)htc/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'htc/batch_normalization_5/batchnorm/addAddV24htc/batch_normalization_5/moments/Squeeze_1:output:02htc/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/RsqrtRsqrt+htc/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:?
'htc/batch_normalization_5/batchnorm/mulMul-htc/batch_normalization_5/batchnorm/Rsqrt:y:07htc/batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/mul_1Mulhtc/dense/Softmax:softmax:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes

: ?
)htc/batch_normalization_5/batchnorm/mul_2Mul2htc/batch_normalization_5/moments/Squeeze:output:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:?
'htc/batch_normalization_5/batchnorm/subSub5htc/batch_normalization_5/Cast/ReadVariableOp:value:0-htc/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
)htc/batch_normalization_5/batchnorm/add_1AddV2-htc/batch_normalization_5/batchnorm/mul_1:z:0+htc/batch_normalization_5/batchnorm/sub:z:0*
T0*
_output_shapes

: y
htc/activation/SoftmaxSoftmax-htc/batch_normalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes

: h
$sparse_categorical_crossentropy/CastCastlabels*

DstT0	*

SrcT0*
_output_shapes
: v
%sparse_categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits-htc/batch_normalization_5/batchnorm/add_1:z:0(sparse_categorical_crossentropy/Cast:y:0*
T0*$
_output_shapes
: : x
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: |
:sparse_categorical_crossentropy/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B : ?
?sparse_categorical_crossentropy/weighted_loss/num_elements/CastCastCsparse_categorical_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: t
2sparse_categorical_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: ?
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: ?
3sparse_categorical_crossentropy/weighted_loss/valueDivNoNan<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsPgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:?????????:??????????
Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanDivNoNanones:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ?
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/SumSumPgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan:z:0\gradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: ?
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/ReshapeReshapeNgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum:output:0Pgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: ?
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/NegNeg<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: ?
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1DivNoNanIgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Neg:y:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ?
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2DivNoNanRgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1:z:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ?
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mulMulones:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: ?
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1SumIgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mul:z:0\gradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: ?
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape_1ReshapePgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: ?
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/ReshapeReshapeRgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape:output:0Tgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: ?
Agradient_tape/sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB ?
@gradient_tape/sparse_categorical_crossentropy/weighted_loss/TileTileLgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape:output:0Jgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: ?
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1ReshapeIgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile:output:0Tgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:?
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1TileNgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1:output:0Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: ?
?gradient_tape/sparse_categorical_crossentropy/weighted_loss/MulMulKgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1:output:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: ?
`gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
\gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims
ExpandDimsCgradient_tape/sparse_categorical_crossentropy/weighted_loss/Mul:z:0igradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dim:output:0*
T0*
_output_shapes

: ?
Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulMulegradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims:output:0rsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:backprop:0*
T0*
_output_shapes

: ?
Pgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB"       ?
Pgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB:?
Mgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgsBroadcastGradientArgsYgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s0:output:0Ygradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:?????????:??????????
Mgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
;gradient_tape/htc/batch_normalization_5/batchnorm/add_1/SumSumYgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0Vgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
Egradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
?gradient_tape/htc/batch_normalization_5/batchnorm/add_1/ReshapeReshapeDgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Sum:output:0Ngradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape/shape:output:0*
T0*
_output_shapes
:?
;gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/MulMulYgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes

: ?
=gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Mul_1Mulhtc/dense/Softmax:softmax:0Ygradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0*
T0*
_output_shapes

: ?
Mgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
;gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/SumSumAgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Mul_1:z:0Vgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
Egradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
?gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/ReshapeReshapeDgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Sum:output:0Ngradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Reshape/shape:output:0*
T0*
_output_shapes
:?
9gradient_tape/htc/batch_normalization_5/batchnorm/sub/NegNegHgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0*
T0*
_output_shapes
:?
;gradient_tape/htc/batch_normalization_5/batchnorm/mul_2/MulMul=gradient_tape/htc/batch_normalization_5/batchnorm/sub/Neg:y:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:?
=gradient_tape/htc/batch_normalization_5/batchnorm/mul_2/Mul_1Mul=gradient_tape/htc/batch_normalization_5/batchnorm/sub/Neg:y:02htc/batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:?
5gradient_tape/htc/batch_normalization_5/moments/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
7gradient_tape/htc/batch_normalization_5/moments/ReshapeReshape?gradient_tape/htc/batch_normalization_5/batchnorm/mul_2/Mul:z:0>gradient_tape/htc/batch_normalization_5/moments/Shape:output:0*
T0*
_output_shapes

:?
AddNAddNHgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Reshape:output:0Agradient_tape/htc/batch_normalization_5/batchnorm/mul_2/Mul_1:z:0*
N*
T0*
_output_shapes
:?
9gradient_tape/htc/batch_normalization_5/batchnorm/mul/MulMul
AddN:sum:07htc/batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:?
;gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1Mul
AddN:sum:0-htc/batch_normalization_5/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:?
9gradient_tape/htc/batch_normalization_5/moments/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"      {
9gradient_tape/htc/batch_normalization_5/moments/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :?
7gradient_tape/htc/batch_normalization_5/moments/MaximumMaximumBgradient_tape/htc/batch_normalization_5/moments/Maximum/x:output:0Bgradient_tape/htc/batch_normalization_5/moments/Maximum/y:output:0*
T0*
_output_shapes
:?
:gradient_tape/htc/batch_normalization_5/moments/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"       ?
8gradient_tape/htc/batch_normalization_5/moments/floordivFloorDivCgradient_tape/htc/batch_normalization_5/moments/floordiv/x:output:0;gradient_tape/htc/batch_normalization_5/moments/Maximum:z:0*
T0*
_output_shapes
:?
?gradient_tape/htc/batch_normalization_5/moments/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
9gradient_tape/htc/batch_normalization_5/moments/Reshape_1Reshape@gradient_tape/htc/batch_normalization_5/moments/Reshape:output:0Hgradient_tape/htc/batch_normalization_5/moments/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
>gradient_tape/htc/batch_normalization_5/moments/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"       ?
4gradient_tape/htc/batch_normalization_5/moments/TileTileBgradient_tape/htc/batch_normalization_5/moments/Reshape_1:output:0Ggradient_tape/htc/batch_normalization_5/moments/Tile/multiples:output:0*
T0*
_output_shapes

: z
5gradient_tape/htc/batch_normalization_5/moments/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
7gradient_tape/htc/batch_normalization_5/moments/truedivRealDiv=gradient_tape/htc/batch_normalization_5/moments/Tile:output:0>gradient_tape/htc/batch_normalization_5/moments/Const:output:0*
T0*
_output_shapes

: ?
;gradient_tape/htc/batch_normalization_5/batchnorm/RsqrtGrad	RsqrtGrad-htc/batch_normalization_5/batchnorm/Rsqrt:y:0=gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul:z:0*
T0*
_output_shapes
:?
7gradient_tape/htc/batch_normalization_5/moments/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
9gradient_tape/htc/batch_normalization_5/moments/Reshape_2Reshape?gradient_tape/htc/batch_normalization_5/batchnorm/RsqrtGrad:z:0@gradient_tape/htc/batch_normalization_5/moments/Shape_1:output:0*
T0*
_output_shapes

:?
?gradient_tape/htc/batch_normalization_5/moments/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
9gradient_tape/htc/batch_normalization_5/moments/Reshape_3ReshapeBgradient_tape/htc/batch_normalization_5/moments/Reshape_2:output:0Hgradient_tape/htc/batch_normalization_5/moments/Reshape_3/shape:output:0*
T0*
_output_shapes

:?
@gradient_tape/htc/batch_normalization_5/moments/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"       ?
6gradient_tape/htc/batch_normalization_5/moments/Tile_1TileBgradient_tape/htc/batch_normalization_5/moments/Reshape_3:output:0Igradient_tape/htc/batch_normalization_5/moments/Tile_1/multiples:output:0*
T0*
_output_shapes

: |
7gradient_tape/htc/batch_normalization_5/moments/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   B?
9gradient_tape/htc/batch_normalization_5/moments/truediv_1RealDiv?gradient_tape/htc/batch_normalization_5/moments/Tile_1:output:0@gradient_tape/htc/batch_normalization_5/moments/Const_1:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes

: ?
6gradient_tape/htc/batch_normalization_5/moments/scalarConst:^gradient_tape/htc/batch_normalization_5/moments/truediv_1*
_output_shapes
: *
dtype0*
valueB
 *   @?
3gradient_tape/htc/batch_normalization_5/moments/MulMul?gradient_tape/htc/batch_normalization_5/moments/scalar:output:0=gradient_tape/htc/batch_normalization_5/moments/truediv_1:z:0*
T0*
_output_shapes

: ?
3gradient_tape/htc/batch_normalization_5/moments/subSubhtc/dense/Softmax:softmax:07htc/batch_normalization_5/moments/StopGradient:output:0:^gradient_tape/htc/batch_normalization_5/moments/truediv_1*
T0*
_output_shapes

: ?
5gradient_tape/htc/batch_normalization_5/moments/mul_1Mul7gradient_tape/htc/batch_normalization_5/moments/Mul:z:07gradient_tape/htc/batch_normalization_5/moments/sub:z:0*
T0*
_output_shapes

: ?
Hgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB"       ?
Hgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB"      ?
Egradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgsBroadcastGradientArgsQgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s0:output:0Qgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:?????????:??????????
AddN_1AddN?gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Mul:z:0;gradient_tape/htc/batch_normalization_5/moments/truediv:z:09gradient_tape/htc/batch_normalization_5/moments/mul_1:z:0*
N*
T0*
_output_shapes

: v
gradient_tape/htc/dense/mulMulAddN_1:sum:0htc/dense/Softmax:softmax:0*
T0*
_output_shapes

: x
-gradient_tape/htc/dense/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gradient_tape/htc/dense/SumSumgradient_tape/htc/dense/mul:z:06gradient_tape/htc/dense/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(
gradient_tape/htc/dense/subSubAddN_1:sum:0$gradient_tape/htc/dense/Sum:output:0*
T0*
_output_shapes

: ?
gradient_tape/htc/dense/mul_1Mulgradient_tape/htc/dense/sub:z:0htc/dense/Softmax:softmax:0*
T0*
_output_shapes

: ?
+gradient_tape/htc/dense/BiasAdd/BiasAddGradBiasAddGrad!gradient_tape/htc/dense/mul_1:z:0*
T0*
_output_shapes
:?
%gradient_tape/htc/dense/MatMul/MatMulMatMul!gradient_tape/htc/dense/mul_1:z:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?$*
transpose_b(?
'gradient_tape/htc/dense/MatMul/MatMul_1MatMulhtc/flatten/Reshape:output:0!gradient_tape/htc/dense/mul_1:z:0*
T0*
_output_shapes
:	?$*
transpose_a(x
gradient_tape/htc/flatten/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!gradient_tape/htc/flatten/ReshapeReshape/gradient_tape/htc/dense/MatMul/MatMul:product:0(gradient_tape/htc/flatten/Shape:output:0*
T0*'
_output_shapes
: ?
/gradient_tape/htc/transpose_1/InvertPermutationInvertPermutationhtc/transpose_1/perm:output:0*
_output_shapes
:?
'gradient_tape/htc/transpose_1/transpose	Transpose*gradient_tape/htc/flatten/Reshape:output:03gradient_tape/htc/transpose_1/InvertPermutation:y:0*
T0*'
_output_shapes
: ?l
'gradient_tape/htc/dropout/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
*gradient_tape/htc/dropout/dropout/SelectV2SelectV2$htc/dropout/dropout/GreaterEqual:z:0+gradient_tape/htc/transpose_1/transpose:y:00gradient_tape/htc/dropout/dropout/zeros:output:0*
T0*'
_output_shapes
: ??
'gradient_tape/htc/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)gradient_tape/htc/dropout/dropout/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
7gradient_tape/htc/dropout/dropout/BroadcastGradientArgsBroadcastGradientArgs0gradient_tape/htc/dropout/dropout/Shape:output:02gradient_tape/htc/dropout/dropout/Shape_1:output:0*2
_output_shapes 
:?????????:??????????
%gradient_tape/htc/dropout/dropout/SumSum3gradient_tape/htc/dropout/dropout/SelectV2:output:0<gradient_tape/htc/dropout/dropout/BroadcastGradientArgs:r0:0*
T0*'
_output_shapes
: ?*
	keep_dims(?
)gradient_tape/htc/dropout/dropout/ReshapeReshape.gradient_tape/htc/dropout/dropout/Sum:output:00gradient_tape/htc/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
: ??
,gradient_tape/htc/dropout/dropout/SelectV2_1SelectV2$htc/dropout/dropout/GreaterEqual:z:00gradient_tape/htc/dropout/dropout/zeros:output:0+gradient_tape/htc/transpose_1/transpose:y:0*
T0*'
_output_shapes
: ?l
)gradient_tape/htc/dropout/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB ?
9gradient_tape/htc/dropout/dropout/BroadcastGradientArgs_1BroadcastGradientArgs2gradient_tape/htc/dropout/dropout/Shape_2:output:02gradient_tape/htc/dropout/dropout/Shape_1:output:0*2
_output_shapes 
:?????????:??????????
'gradient_tape/htc/dropout/dropout/Sum_1Sum5gradient_tape/htc/dropout/dropout/SelectV2_1:output:0>gradient_tape/htc/dropout/dropout/BroadcastGradientArgs_1:r0:0*
T0*&
_output_shapes
:*
	keep_dims(?
+gradient_tape/htc/dropout/dropout/Reshape_1Reshape0gradient_tape/htc/dropout/dropout/Sum_1:output:02gradient_tape/htc/dropout/dropout/Shape_2:output:0*
T0*
_output_shapes
: ?
%gradient_tape/htc/dropout/dropout/MulMul2gradient_tape/htc/dropout/dropout/Reshape:output:0"htc/dropout/dropout/Const:output:0*
T0*'
_output_shapes
: ?{
-gradient_tape/htc/transpose/InvertPermutationInvertPermutationhtc/transpose/perm:output:0*
_output_shapes
:?
%gradient_tape/htc/transpose/transpose	Transpose)gradient_tape/htc/dropout/dropout/Mul:z:01gradient_tape/htc/transpose/InvertPermutation:y:0*
T0*'
_output_shapes
: ??
"gradient_tape/htc/re_lu_4/ReluGradReluGrad)gradient_tape/htc/transpose/transpose:y:0htc/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
: ?T
zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_1Const*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_2Const*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_3Const*
_output_shapes	
:?*
dtype0*
valueB?*    x

zeros_like	ZerosLike<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:?
<gradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_4/ReluGrad:backprops:0$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:0<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*=
_output_shapes+
): ?:?:?: : *
epsilon%o?:?
5gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_4/BiasAdd:output:0$htc/max_pooling2d_4/MaxPool:output:0Igradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:x_backprop:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
.gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes	
:??
(gradient_tape/htc/conv2d_4/Conv2D/ShapeNShapeNhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::?
5gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_4/Conv2D/ShapeN:output:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGrad:output:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
6gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu_3/Relu:activations:01gradient_tape/htc/conv2d_4/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGrad:output:0*
T0*(
_output_shapes
:??*
paddingVALID*
strides
?
"gradient_tape/htc/re_lu_3/ReluGradReluGrad>gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropInput:output:0htc/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
: ?V
zeros_4Const*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_5Const*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_6Const*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_7Const*
_output_shapes	
:?*
dtype0*
valueB?*    z
zeros_like_1	ZerosLike<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:?
<gradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_3/ReluGrad:backprops:0$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:0<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*=
_output_shapes+
): ?:?:?: : *
epsilon%o?:?
5gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_3/BiasAdd:output:0$htc/max_pooling2d_3/MaxPool:output:0Igradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:x_backprop:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
.gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes	
:??
(gradient_tape/htc/conv2d_3/Conv2D/ShapeNShapeNhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::?
5gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_3/Conv2D/ShapeN:output:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGrad:output:0*
T0*'
_output_shapes
: ?*
paddingVALID*
strides
?
6gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu_2/Relu:activations:01gradient_tape/htc/conv2d_3/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGrad:output:0*
T0*(
_output_shapes
:??*
paddingVALID*
strides
?
"gradient_tape/htc/re_lu_2/ReluGradReluGrad>gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropInput:output:0htc/re_lu_2/Relu:activations:0*
T0*'
_output_shapes
: ?V
zeros_8Const*
_output_shapes	
:?*
dtype0*
valueB?*    V
zeros_9Const*
_output_shapes	
:?*
dtype0*
valueB?*    W
zeros_10Const*
_output_shapes	
:?*
dtype0*
valueB?*    W
zeros_11Const*
_output_shapes	
:?*
dtype0*
valueB?*    z
zeros_like_2	ZerosLike<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:?
<gradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_2/ReluGrad:backprops:0$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:0<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*=
_output_shapes+
): ?:?:?: : *
epsilon%o?:?
5gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_2/BiasAdd:output:0$htc/max_pooling2d_2/MaxPool:output:0Igradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:x_backprop:0*'
_output_shapes
: ?*
ksize
*
paddingVALID*
strides
?
.gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes	
:??
(gradient_tape/htc/conv2d_2/Conv2D/ShapeNShapeNhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::?
5gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_2/Conv2D/ShapeN:output:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: @*
paddingVALID*
strides
?
6gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu_1/Relu:activations:01gradient_tape/htc/conv2d_2/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGrad:output:0*
T0*'
_output_shapes
:@?*
paddingVALID*
strides
?
"gradient_tape/htc/re_lu_1/ReluGradReluGrad>gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropInput:output:0htc/re_lu_1/Relu:activations:0*
T0*&
_output_shapes
: @U
zeros_12Const*
_output_shapes
:@*
dtype0*
valueB@*    U
zeros_13Const*
_output_shapes
:@*
dtype0*
valueB@*    U
zeros_14Const*
_output_shapes
:@*
dtype0*
valueB@*    U
zeros_15Const*
_output_shapes
:@*
dtype0*
valueB@*    z
zeros_like_3	ZerosLike<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:?
<gradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_1/ReluGrad:backprops:0$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:0<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*:
_output_shapes(
&: @:@:@: : *
epsilon%o?:?
5gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_1/BiasAdd:output:0$htc/max_pooling2d_1/MaxPool:output:0Igradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:x_backprop:0*&
_output_shapes
: ^^@*
ksize
*
paddingVALID*
strides
?
.gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes
:@?
(gradient_tape/htc/conv2d_1/Conv2D/ShapeNShapeNhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::?
5gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_1/Conv2D/ShapeN:output:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: bb *
paddingVALID*
strides
?
6gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu/Relu:activations:01gradient_tape/htc/conv2d_1/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: @*
paddingVALID*
strides
?
 gradient_tape/htc/re_lu/ReluGradReluGrad>gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropInput:output:0htc/re_lu/Relu:activations:0*
T0*&
_output_shapes
: bb U
zeros_16Const*
_output_shapes
: *
dtype0*
valueB *    U
zeros_17Const*
_output_shapes
: *
dtype0*
valueB *    U
zeros_18Const*
_output_shapes
: *
dtype0*
valueB *    U
zeros_19Const*
_output_shapes
: *
dtype0*
valueB *    x
zeros_like_4	ZerosLike:htc/batch_normalization/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:?
:gradient_tape/htc/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3,gradient_tape/htc/re_lu/ReluGrad:backprops:0"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:0:htc/batch_normalization/FusedBatchNormV3:reserve_space_1:0:htc/batch_normalization/FusedBatchNormV3:reserve_space_2:0:htc/batch_normalization/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*:
_output_shapes(
&: bb : : : : *
epsilon%o?:?
3gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d/BiasAdd:output:0"htc/max_pooling2d/MaxPool:output:0Ggradient_tape/htc/batch_normalization/FusedBatchNormGradV3:x_backprop:0*(
_output_shapes
: ?? *
ksize
*
paddingVALID*
strides
?
,gradient_tape/htc/conv2d/BiasAdd/BiasAddGradBiasAddGrad<gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes
: ?
&gradient_tape/htc/conv2d/Conv2D/ShapeNShapeNhtc/Cast:y:0(htc/conv2d/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::?
3gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropInputConv2DBackpropInput/gradient_tape/htc/conv2d/Conv2D/ShapeN:output:0(htc/conv2d/Conv2D/ReadVariableOp:value:0<gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGrad:output:0*
T0*(
_output_shapes
: ??*
paddingVALID*
strides
?
4gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/Cast:y:0/gradient_tape/htc/conv2d/Conv2D/ShapeN:output:1<gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
IdentityIdentity=gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
: r

Identity_1Identity5gradient_tape/htc/conv2d/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
: ?

Identity_2IdentityKgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes
: ?

Identity_3IdentityLgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes
: ?

Identity_4Identity?gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
: @t

Identity_5Identity7gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:@?

Identity_6IdentityMgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes
:@?

Identity_7IdentityNgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes
:@?

Identity_8Identity?gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilter:output:0*
T0*'
_output_shapes
:@?u

Identity_9Identity7gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:??
Identity_10IdentityMgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes	
:??
Identity_11IdentityNgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes	
:??
Identity_12Identity?gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilter:output:0*
T0*(
_output_shapes
:??v
Identity_13Identity7gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:??
Identity_14IdentityMgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes	
:??
Identity_15IdentityNgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes	
:??
Identity_16Identity?gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilter:output:0*
T0*(
_output_shapes
:??v
Identity_17Identity7gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:??
Identity_18IdentityMgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes	
:??
Identity_19IdentityNgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes	
:?t
Identity_20Identity1gradient_tape/htc/dense/MatMul/MatMul_1:product:0*
T0*
_output_shapes
:	?$r
Identity_21Identity4gradient_tape/htc/dense/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:}
Identity_22Identity?gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1:z:0*
T0*
_output_shapes
:?
Identity_23IdentityHgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0*
T0*
_output_shapes
:?
	IdentityN	IdentityN=gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilter:output:05gradient_tape/htc/conv2d/BiasAdd/BiasAddGrad:output:0Kgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:scale_backprop:0Lgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:offset_backprop:01gradient_tape/htc/dense/MatMul/MatMul_1:product:04gradient_tape/htc/dense/BiasAdd/BiasAddGrad:output:0?gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1:z:0Hgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0=gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilter:output:05gradient_tape/htc/conv2d/BiasAdd/BiasAddGrad:output:0Kgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:scale_backprop:0Lgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:offset_backprop:01gradient_tape/htc/dense/MatMul/MatMul_1:product:04gradient_tape/htc/dense/BiasAdd/BiasAddGrad:output:0?gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1:z:0Hgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0*9
T4
220*+
_gradient_op_typeCustomGradient-46326*?
_output_shapes?
?: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:	?$:::: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:	?$:::?
StatefulPartitionedCallStatefulPartitionedCallIdentityN:output:0)htc_conv2d_conv2d_readvariableop_resourceunknown	unknown_0	unknown_1	unknown_2!^htc/conv2d/Conv2D/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46438?
StatefulPartitionedCall_1StatefulPartitionedCallIdentityN:output:1*htc_conv2d_biasadd_readvariableop_resourceunknown	unknown_0	unknown_3	unknown_4"^htc/conv2d/BiasAdd/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46487?
StatefulPartitionedCall_2StatefulPartitionedCallIdentityN:output:2/htc_batch_normalization_readvariableop_resourceunknown	unknown_0	unknown_5	unknown_6'^htc/batch_normalization/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46534?
StatefulPartitionedCall_3StatefulPartitionedCallIdentityN:output:31htc_batch_normalization_readvariableop_1_resourceunknown	unknown_0	unknown_7	unknown_8)^htc/batch_normalization/ReadVariableOp_1*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46581?
StatefulPartitionedCall_4StatefulPartitionedCallIdentityN:output:4+htc_conv2d_1_conv2d_readvariableop_resourceunknown	unknown_0	unknown_9
unknown_10#^htc/conv2d_1/Conv2D/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46628?
StatefulPartitionedCall_5StatefulPartitionedCallIdentityN:output:5,htc_conv2d_1_biasadd_readvariableop_resourceunknown	unknown_0
unknown_11
unknown_12$^htc/conv2d_1/BiasAdd/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46675?
StatefulPartitionedCall_6StatefulPartitionedCallIdentityN:output:61htc_batch_normalization_1_readvariableop_resourceunknown	unknown_0
unknown_13
unknown_14)^htc/batch_normalization_1/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46722?
StatefulPartitionedCall_7StatefulPartitionedCallIdentityN:output:73htc_batch_normalization_1_readvariableop_1_resourceunknown	unknown_0
unknown_15
unknown_16+^htc/batch_normalization_1/ReadVariableOp_1*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46769?
StatefulPartitionedCall_8StatefulPartitionedCallIdentityN:output:8+htc_conv2d_2_conv2d_readvariableop_resourceunknown	unknown_0
unknown_17
unknown_18#^htc/conv2d_2/Conv2D/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46816?
StatefulPartitionedCall_9StatefulPartitionedCallIdentityN:output:9,htc_conv2d_2_biasadd_readvariableop_resourceunknown	unknown_0
unknown_19
unknown_20$^htc/conv2d_2/BiasAdd/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46863?
StatefulPartitionedCall_10StatefulPartitionedCallIdentityN:output:101htc_batch_normalization_2_readvariableop_resourceunknown	unknown_0
unknown_21
unknown_22)^htc/batch_normalization_2/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46910?
StatefulPartitionedCall_11StatefulPartitionedCallIdentityN:output:113htc_batch_normalization_2_readvariableop_1_resourceunknown	unknown_0
unknown_23
unknown_24+^htc/batch_normalization_2/ReadVariableOp_1*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_46957?
StatefulPartitionedCall_12StatefulPartitionedCallIdentityN:output:12+htc_conv2d_3_conv2d_readvariableop_resourceunknown	unknown_0
unknown_25
unknown_26#^htc/conv2d_3/Conv2D/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47004?
StatefulPartitionedCall_13StatefulPartitionedCallIdentityN:output:13,htc_conv2d_3_biasadd_readvariableop_resourceunknown	unknown_0
unknown_27
unknown_28$^htc/conv2d_3/BiasAdd/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47051?
StatefulPartitionedCall_14StatefulPartitionedCallIdentityN:output:141htc_batch_normalization_3_readvariableop_resourceunknown	unknown_0
unknown_29
unknown_30)^htc/batch_normalization_3/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47098?
StatefulPartitionedCall_15StatefulPartitionedCallIdentityN:output:153htc_batch_normalization_3_readvariableop_1_resourceunknown	unknown_0
unknown_31
unknown_32+^htc/batch_normalization_3/ReadVariableOp_1*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47145?
StatefulPartitionedCall_16StatefulPartitionedCallIdentityN:output:16+htc_conv2d_4_conv2d_readvariableop_resourceunknown	unknown_0
unknown_33
unknown_34#^htc/conv2d_4/Conv2D/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47192?
StatefulPartitionedCall_17StatefulPartitionedCallIdentityN:output:17,htc_conv2d_4_biasadd_readvariableop_resourceunknown	unknown_0
unknown_35
unknown_36$^htc/conv2d_4/BiasAdd/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47239?
StatefulPartitionedCall_18StatefulPartitionedCallIdentityN:output:181htc_batch_normalization_4_readvariableop_resourceunknown	unknown_0
unknown_37
unknown_38)^htc/batch_normalization_4/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47286?
StatefulPartitionedCall_19StatefulPartitionedCallIdentityN:output:193htc_batch_normalization_4_readvariableop_1_resourceunknown	unknown_0
unknown_39
unknown_40+^htc/batch_normalization_4/ReadVariableOp_1*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47333?
StatefulPartitionedCall_20StatefulPartitionedCallIdentityN:output:20(htc_dense_matmul_readvariableop_resourceunknown	unknown_0
unknown_41
unknown_42 ^htc/dense/MatMul/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47380?
StatefulPartitionedCall_21StatefulPartitionedCallIdentityN:output:21)htc_dense_biasadd_readvariableop_resourceunknown	unknown_0
unknown_43
unknown_44!^htc/dense/BiasAdd/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47427?
StatefulPartitionedCall_22StatefulPartitionedCallIdentityN:output:228htc_batch_normalization_5_cast_1_readvariableop_resourceunknown	unknown_0
unknown_45
unknown_460^htc/batch_normalization_5/Cast_1/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47474?
StatefulPartitionedCall_23StatefulPartitionedCallIdentityN:output:236htc_batch_normalization_5_cast_readvariableop_resourceunknown	unknown_0
unknown_47
unknown_48.^htc/batch_normalization_5/Cast/ReadVariableOp*
Tin

2*

Tout
 *
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes
 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__update_step_xla_47521G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R?
AssignAddVariableOpAssignAddVariableOpunknownConst:output:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_10^StatefulPartitionedCall_11^StatefulPartitionedCall_12^StatefulPartitionedCall_13^StatefulPartitionedCall_14^StatefulPartitionedCall_15^StatefulPartitionedCall_16^StatefulPartitionedCall_17^StatefulPartitionedCall_18^StatefulPartitionedCall_19^StatefulPartitionedCall_2^StatefulPartitionedCall_20^StatefulPartitionedCall_21^StatefulPartitionedCall_22^StatefulPartitionedCall_23^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6^StatefulPartitionedCall_7^StatefulPartitionedCall_8^StatefulPartitionedCall_9*
_output_shapes
 *
dtype0	F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: ?
SumSum7sparse_categorical_crossentropy/weighted_loss/value:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceCast:y:0^AssignAddVariableOp_1*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1^AssignAddVariableOp_2*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: H
Identity_24Identitydiv_no_nan:z:0*
T0*
_output_shapes
: J
Cast_1Castlabels*

DstT0*

SrcT0*
_output_shapes
: O
ShapeConst*
_output_shapes
:*
dtype0*
valueB: [
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????r
ArgMaxArgMax htc/activation/Softmax:softmax:0ArgMax/dimension:output:0*
T0*
_output_shapes
: S
Cast_2CastArgMax:output:0*

DstT0*

SrcT0	*
_output_shapes
: K
EqualEqual
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: M
Cast_3Cast	Equal:z:0*

DstT0*

SrcT0
*
_output_shapes
: Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: s
Sum_1Sum
Cast_3:y:0Const_1:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: ?
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resourceSum_1:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0H
Size_1Const*
_output_shapes
: *
dtype0*
value	B : O
Cast_4CastSize_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_4AssignAddVariableOpassignaddvariableop_4_resource
Cast_4:y:0^AssignAddVariableOp_3*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0?
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3^AssignAddVariableOp_4*
_output_shapes
: *
dtype0?
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_4_resource^AssignAddVariableOp_4*
_output_shapes
: *
dtype0?
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: J
Identity_25Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: *(
_construction_contextkEagerRuntime*?
_input_shapes?
?: ??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32.
AssignAddVariableOp_4AssignAddVariableOp_422
StatefulPartitionedCallStatefulPartitionedCall26
StatefulPartitionedCall_1StatefulPartitionedCall_128
StatefulPartitionedCall_10StatefulPartitionedCall_1028
StatefulPartitionedCall_11StatefulPartitionedCall_1128
StatefulPartitionedCall_12StatefulPartitionedCall_1228
StatefulPartitionedCall_13StatefulPartitionedCall_1328
StatefulPartitionedCall_14StatefulPartitionedCall_1428
StatefulPartitionedCall_15StatefulPartitionedCall_1528
StatefulPartitionedCall_16StatefulPartitionedCall_1628
StatefulPartitionedCall_17StatefulPartitionedCall_1728
StatefulPartitionedCall_18StatefulPartitionedCall_1828
StatefulPartitionedCall_19StatefulPartitionedCall_1926
StatefulPartitionedCall_2StatefulPartitionedCall_228
StatefulPartitionedCall_20StatefulPartitionedCall_2028
StatefulPartitionedCall_21StatefulPartitionedCall_2128
StatefulPartitionedCall_22StatefulPartitionedCall_2228
StatefulPartitionedCall_23StatefulPartitionedCall_2326
StatefulPartitionedCall_3StatefulPartitionedCall_326
StatefulPartitionedCall_4StatefulPartitionedCall_426
StatefulPartitionedCall_5StatefulPartitionedCall_526
StatefulPartitionedCall_6StatefulPartitionedCall_626
StatefulPartitionedCall_7StatefulPartitionedCall_726
StatefulPartitionedCall_8StatefulPartitionedCall_826
StatefulPartitionedCall_9StatefulPartitionedCall_926
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_12P
&htc/batch_normalization/AssignNewValue&htc/batch_normalization/AssignNewValue2T
(htc/batch_normalization/AssignNewValue_1(htc/batch_normalization/AssignNewValue_12r
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp2v
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_19htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_12P
&htc/batch_normalization/ReadVariableOp&htc/batch_normalization/ReadVariableOp2T
(htc/batch_normalization/ReadVariableOp_1(htc/batch_normalization/ReadVariableOp_12T
(htc/batch_normalization_1/AssignNewValue(htc/batch_normalization_1/AssignNewValue2X
*htc/batch_normalization_1/AssignNewValue_1*htc/batch_normalization_1/AssignNewValue_12v
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_1/ReadVariableOp(htc/batch_normalization_1/ReadVariableOp2X
*htc/batch_normalization_1/ReadVariableOp_1*htc/batch_normalization_1/ReadVariableOp_12T
(htc/batch_normalization_2/AssignNewValue(htc/batch_normalization_2/AssignNewValue2X
*htc/batch_normalization_2/AssignNewValue_1*htc/batch_normalization_2/AssignNewValue_12v
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_2/ReadVariableOp(htc/batch_normalization_2/ReadVariableOp2X
*htc/batch_normalization_2/ReadVariableOp_1*htc/batch_normalization_2/ReadVariableOp_12T
(htc/batch_normalization_3/AssignNewValue(htc/batch_normalization_3/AssignNewValue2X
*htc/batch_normalization_3/AssignNewValue_1*htc/batch_normalization_3/AssignNewValue_12v
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_3/ReadVariableOp(htc/batch_normalization_3/ReadVariableOp2X
*htc/batch_normalization_3/ReadVariableOp_1*htc/batch_normalization_3/ReadVariableOp_12T
(htc/batch_normalization_4/AssignNewValue(htc/batch_normalization_4/AssignNewValue2X
*htc/batch_normalization_4/AssignNewValue_1*htc/batch_normalization_4/AssignNewValue_12v
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2z
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12T
(htc/batch_normalization_4/ReadVariableOp(htc/batch_normalization_4/ReadVariableOp2X
*htc/batch_normalization_4/ReadVariableOp_1*htc/batch_normalization_4/ReadVariableOp_12V
)htc/batch_normalization_5/AssignMovingAvg)htc/batch_normalization_5/AssignMovingAvg2t
8htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp8htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp2Z
+htc/batch_normalization_5/AssignMovingAvg_1+htc/batch_normalization_5/AssignMovingAvg_12x
:htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2^
-htc/batch_normalization_5/Cast/ReadVariableOp-htc/batch_normalization_5/Cast/ReadVariableOp2b
/htc/batch_normalization_5/Cast_1/ReadVariableOp/htc/batch_normalization_5/Cast_1/ReadVariableOp2F
!htc/conv2d/BiasAdd/ReadVariableOp!htc/conv2d/BiasAdd/ReadVariableOp2D
 htc/conv2d/Conv2D/ReadVariableOp htc/conv2d/Conv2D/ReadVariableOp2J
#htc/conv2d_1/BiasAdd/ReadVariableOp#htc/conv2d_1/BiasAdd/ReadVariableOp2H
"htc/conv2d_1/Conv2D/ReadVariableOp"htc/conv2d_1/Conv2D/ReadVariableOp2J
#htc/conv2d_2/BiasAdd/ReadVariableOp#htc/conv2d_2/BiasAdd/ReadVariableOp2H
"htc/conv2d_2/Conv2D/ReadVariableOp"htc/conv2d_2/Conv2D/ReadVariableOp2J
#htc/conv2d_3/BiasAdd/ReadVariableOp#htc/conv2d_3/BiasAdd/ReadVariableOp2H
"htc/conv2d_3/Conv2D/ReadVariableOp"htc/conv2d_3/Conv2D/ReadVariableOp2J
#htc/conv2d_4/BiasAdd/ReadVariableOp#htc/conv2d_4/BiasAdd/ReadVariableOp2H
"htc/conv2d_4/Conv2D/ReadVariableOp"htc/conv2d_4/Conv2D/ReadVariableOp2D
 htc/dense/BiasAdd/ReadVariableOp htc/dense/BiasAdd/ReadVariableOp2B
htc/dense/MatMul/ReadVariableOphtc/dense/MatMul/ReadVariableOp:P L
(
_output_shapes
: ??
 
_user_specified_nameimages:B>

_output_shapes
: 
 
_user_specified_namelabels
?
F
*__inference_activation_layer_call_fn_48752

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_45008`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_4_layer_call_fn_48604

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_44956i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44586

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_45128

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_44975

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48296

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_44637

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_48619

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45128x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_2_layer_call_fn_48348

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44510?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_1_layer_call_fn_48247

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44434?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
C
'__inference_re_lu_2_layer_call_fn_48402

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_44890i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_re_lu_layer_call_fn_48200

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_44824h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????bb "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????bb :W S
/
_output_shapes
:?????????bb 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48278

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44836

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????^^@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????bb : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????bb 
 
_user_specified_nameinputs
?
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_48407

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_47145
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44541

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?^
?
"__inference_internal_grad_fn_49096
result_grads_0
result_grads_1
result_grads_2
result_grads_3
result_grads_4
result_grads_5
result_grads_6
result_grads_7
result_grads_8
result_grads_9
result_grads_10
result_grads_11
result_grads_12
result_grads_13
result_grads_14
result_grads_15
result_grads_16
result_grads_17
result_grads_18
result_grads_19
result_grads_20
result_grads_21
result_grads_22
result_grads_23
result_grads_24
result_grads_25
result_grads_26
result_grads_27
result_grads_28
result_grads_29
result_grads_30
result_grads_31
result_grads_32
result_grads_33
result_grads_34
result_grads_35
result_grads_36
result_grads_37
result_grads_38
result_grads_39
result_grads_40
result_grads_41
result_grads_42
result_grads_43
result_grads_44
result_grads_45
result_grads_46
result_grads_47
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43
identity_44
identity_45
identity_46
identity_47U
IdentityIdentityresult_grads_0*
T0*&
_output_shapes
: K

Identity_1Identityresult_grads_1*
T0*
_output_shapes
: K

Identity_2Identityresult_grads_2*
T0*
_output_shapes
: K

Identity_3Identityresult_grads_3*
T0*
_output_shapes
: W

Identity_4Identityresult_grads_4*
T0*&
_output_shapes
: @K

Identity_5Identityresult_grads_5*
T0*
_output_shapes
:@K

Identity_6Identityresult_grads_6*
T0*
_output_shapes
:@K

Identity_7Identityresult_grads_7*
T0*
_output_shapes
:@X

Identity_8Identityresult_grads_8*
T0*'
_output_shapes
:@?L

Identity_9Identityresult_grads_9*
T0*
_output_shapes	
:?N
Identity_10Identityresult_grads_10*
T0*
_output_shapes	
:?N
Identity_11Identityresult_grads_11*
T0*
_output_shapes	
:?[
Identity_12Identityresult_grads_12*
T0*(
_output_shapes
:??N
Identity_13Identityresult_grads_13*
T0*
_output_shapes	
:?N
Identity_14Identityresult_grads_14*
T0*
_output_shapes	
:?N
Identity_15Identityresult_grads_15*
T0*
_output_shapes	
:?[
Identity_16Identityresult_grads_16*
T0*(
_output_shapes
:??N
Identity_17Identityresult_grads_17*
T0*
_output_shapes	
:?N
Identity_18Identityresult_grads_18*
T0*
_output_shapes	
:?N
Identity_19Identityresult_grads_19*
T0*
_output_shapes	
:?R
Identity_20Identityresult_grads_20*
T0*
_output_shapes
:	?$M
Identity_21Identityresult_grads_21*
T0*
_output_shapes
:M
Identity_22Identityresult_grads_22*
T0*
_output_shapes
:M
Identity_23Identityresult_grads_23*
T0*
_output_shapes
:?

	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_10result_grads_11result_grads_12result_grads_13result_grads_14result_grads_15result_grads_16result_grads_17result_grads_18result_grads_19result_grads_20result_grads_21result_grads_22result_grads_23result_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_10result_grads_11result_grads_12result_grads_13result_grads_14result_grads_15result_grads_16result_grads_17result_grads_18result_grads_19result_grads_20result_grads_21result_grads_22result_grads_23*9
T4
220*+
_gradient_op_typeCustomGradient-48999*?
_output_shapes?
?: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:	?$:::: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:	?$:::\
Identity_24IdentityIdentityN:output:0*
T0*&
_output_shapes
: P
Identity_25IdentityIdentityN:output:1*
T0*
_output_shapes
: P
Identity_26IdentityIdentityN:output:2*
T0*
_output_shapes
: P
Identity_27IdentityIdentityN:output:3*
T0*
_output_shapes
: \
Identity_28IdentityIdentityN:output:4*
T0*&
_output_shapes
: @P
Identity_29IdentityIdentityN:output:5*
T0*
_output_shapes
:@P
Identity_30IdentityIdentityN:output:6*
T0*
_output_shapes
:@P
Identity_31IdentityIdentityN:output:7*
T0*
_output_shapes
:@]
Identity_32IdentityIdentityN:output:8*
T0*'
_output_shapes
:@?Q
Identity_33IdentityIdentityN:output:9*
T0*
_output_shapes	
:?R
Identity_34IdentityIdentityN:output:10*
T0*
_output_shapes	
:?R
Identity_35IdentityIdentityN:output:11*
T0*
_output_shapes	
:?_
Identity_36IdentityIdentityN:output:12*
T0*(
_output_shapes
:??R
Identity_37IdentityIdentityN:output:13*
T0*
_output_shapes	
:?R
Identity_38IdentityIdentityN:output:14*
T0*
_output_shapes	
:?R
Identity_39IdentityIdentityN:output:15*
T0*
_output_shapes	
:?_
Identity_40IdentityIdentityN:output:16*
T0*(
_output_shapes
:??R
Identity_41IdentityIdentityN:output:17*
T0*
_output_shapes	
:?R
Identity_42IdentityIdentityN:output:18*
T0*
_output_shapes	
:?R
Identity_43IdentityIdentityN:output:19*
T0*
_output_shapes	
:?V
Identity_44IdentityIdentityN:output:20*
T0*
_output_shapes
:	?$Q
Identity_45IdentityIdentityN:output:21*
T0*
_output_shapes
:Q
Identity_46IdentityIdentityN:output:22*
T0*
_output_shapes
:Q
Identity_47IdentityIdentityN:output:23*
T0*
_output_shapes
:"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"#
identity_44Identity_44:output:0"#
identity_45Identity_45:output:0"#
identity_46Identity_46:output:0"#
identity_47Identity_47:output:0*?
_input_shapes?
?: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:	?$:::: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:	?$::::V R
&
_output_shapes
: 
(
_user_specified_nameresult_grads_0:JF

_output_shapes
: 
(
_user_specified_nameresult_grads_1:JF

_output_shapes
: 
(
_user_specified_nameresult_grads_2:JF

_output_shapes
: 
(
_user_specified_nameresult_grads_3:VR
&
_output_shapes
: @
(
_user_specified_nameresult_grads_4:JF

_output_shapes
:@
(
_user_specified_nameresult_grads_5:JF

_output_shapes
:@
(
_user_specified_nameresult_grads_6:JF

_output_shapes
:@
(
_user_specified_nameresult_grads_7:WS
'
_output_shapes
:@?
(
_user_specified_nameresult_grads_8:K	G

_output_shapes	
:?
(
_user_specified_nameresult_grads_9:L
H

_output_shapes	
:?
)
_user_specified_nameresult_grads_10:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_11:YU
(
_output_shapes
:??
)
_user_specified_nameresult_grads_12:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_13:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_14:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_15:YU
(
_output_shapes
:??
)
_user_specified_nameresult_grads_16:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_17:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_18:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_19:PL

_output_shapes
:	?$
)
_user_specified_nameresult_grads_20:KG

_output_shapes
:
)
_user_specified_nameresult_grads_21:KG

_output_shapes
:
)
_user_specified_nameresult_grads_22:KG

_output_shapes
:
)
_user_specified_nameresult_grads_23:WS
&
_output_shapes
: 
)
_user_specified_nameresult_grads_24:KG

_output_shapes
: 
)
_user_specified_nameresult_grads_25:KG

_output_shapes
: 
)
_user_specified_nameresult_grads_26:KG

_output_shapes
: 
)
_user_specified_nameresult_grads_27:WS
&
_output_shapes
: @
)
_user_specified_nameresult_grads_28:KG

_output_shapes
:@
)
_user_specified_nameresult_grads_29:KG

_output_shapes
:@
)
_user_specified_nameresult_grads_30:KG

_output_shapes
:@
)
_user_specified_nameresult_grads_31:X T
'
_output_shapes
:@?
)
_user_specified_nameresult_grads_32:L!H

_output_shapes	
:?
)
_user_specified_nameresult_grads_33:L"H

_output_shapes	
:?
)
_user_specified_nameresult_grads_34:L#H

_output_shapes	
:?
)
_user_specified_nameresult_grads_35:Y$U
(
_output_shapes
:??
)
_user_specified_nameresult_grads_36:L%H

_output_shapes	
:?
)
_user_specified_nameresult_grads_37:L&H

_output_shapes	
:?
)
_user_specified_nameresult_grads_38:L'H

_output_shapes	
:?
)
_user_specified_nameresult_grads_39:Y(U
(
_output_shapes
:??
)
_user_specified_nameresult_grads_40:L)H

_output_shapes	
:?
)
_user_specified_nameresult_grads_41:L*H

_output_shapes	
:?
)
_user_specified_nameresult_grads_42:L+H

_output_shapes	
:?
)
_user_specified_nameresult_grads_43:P,L

_output_shapes
:	?$
)
_user_specified_nameresult_grads_44:K-G

_output_shapes
:
)
_user_specified_nameresult_grads_45:K.G

_output_shapes
:
)
_user_specified_nameresult_grads_46:K/G

_output_shapes
:
)
_user_specified_nameresult_grads_47
? 
?
"__inference__update_step_xla_47051
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?	
?
5__inference_batch_normalization_3_layer_call_fn_48449

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44586?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?t
?
>__inference_htc_layer_call_and_return_conditional_losses_45757
input_1&
conv2d_45655: 
conv2d_45657: '
batch_normalization_45661: '
batch_normalization_45663: '
batch_normalization_45665: '
batch_normalization_45667: (
conv2d_1_45671: @
conv2d_1_45673:@)
batch_normalization_1_45677:@)
batch_normalization_1_45679:@)
batch_normalization_1_45681:@)
batch_normalization_1_45683:@)
conv2d_2_45687:@?
conv2d_2_45689:	?*
batch_normalization_2_45693:	?*
batch_normalization_2_45695:	?*
batch_normalization_2_45697:	?*
batch_normalization_2_45699:	?*
conv2d_3_45703:??
conv2d_3_45705:	?*
batch_normalization_3_45709:	?*
batch_normalization_3_45711:	?*
batch_normalization_3_45713:	?*
batch_normalization_3_45715:	?*
conv2d_4_45719:??
conv2d_4_45721:	?*
batch_normalization_4_45725:	?*
batch_normalization_4_45727:	?*
batch_normalization_4_45729:	?*
batch_normalization_4_45731:	?
dense_45741:	?$
dense_45743:)
batch_normalization_5_45746:)
batch_normalization_5_45748:)
batch_normalization_5_45750:)
batch_normalization_5_45752:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_45655conv2d_45657*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44803?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44333?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_45661batch_normalization_45663batch_normalization_45665batch_normalization_45667*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44389?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_44824?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_45671conv2d_1_45673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44836?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44409?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_45677batch_normalization_1_45679batch_normalization_1_45681batch_normalization_1_45683*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44465?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_44857?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_45687conv2d_2_45689*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44869?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44485?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_45693batch_normalization_2_45695batch_normalization_2_45697batch_normalization_2_45699*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44541?
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_44890?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_45703conv2d_3_45705*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44902?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44561?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_45709batch_normalization_3_45711batch_normalization_3_45713batch_normalization_3_45715*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44617?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_44923?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_45719conv2d_4_45721*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44935?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_44637?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_45725batch_normalization_4_45727batch_normalization_4_45729batch_normalization_4_45731*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44693?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_44956g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:???????????
dropout/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45128i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_1	Transpose(dropout/StatefulPartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:???????????
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44975?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_45741dense_45743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44988?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_45746batch_normalization_5_45748batch_normalization_5_45750batch_normalization_5_45752*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44775?
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_45008r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
5__inference_batch_normalization_5_layer_call_fn_48680

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48426

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_48436

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48234

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_htc_layer_call_fn_45086
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?

unknown_29:	?$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_htc_layer_call_and_return_conditional_losses_45011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
? 
?
"__inference__update_step_xla_46769
gradient
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:@+
sub_3_readvariableop_resource:@??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:@*
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:@?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
:@n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:@*
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:@?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:@*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:@?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:@*
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:@O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:@f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:@: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
? 
?
"__inference__update_step_xla_47427
gradient
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:+
sub_3_readvariableop_resource:??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:*
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
:n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:*
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:*
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48195

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_48647

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_46957
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
#__inference_htc_layer_call_fn_45547
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?

unknown_29:	?$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
 #$*2
config_proto" 

CPU

GPU2 *0J 8? *G
fBR@
>__inference_htc_layer_call_and_return_conditional_losses_45395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44409

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44333

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_48624

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_1_layer_call_fn_48301

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_44857h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_48757

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_48159

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44389?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_44965

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44693

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?s
?
>__inference_htc_layer_call_and_return_conditional_losses_45652
input_1&
conv2d_45550: 
conv2d_45552: '
batch_normalization_45556: '
batch_normalization_45558: '
batch_normalization_45560: '
batch_normalization_45562: (
conv2d_1_45566: @
conv2d_1_45568:@)
batch_normalization_1_45572:@)
batch_normalization_1_45574:@)
batch_normalization_1_45576:@)
batch_normalization_1_45578:@)
conv2d_2_45582:@?
conv2d_2_45584:	?*
batch_normalization_2_45588:	?*
batch_normalization_2_45590:	?*
batch_normalization_2_45592:	?*
batch_normalization_2_45594:	?*
conv2d_3_45598:??
conv2d_3_45600:	?*
batch_normalization_3_45604:	?*
batch_normalization_3_45606:	?*
batch_normalization_3_45608:	?*
batch_normalization_3_45610:	?*
conv2d_4_45614:??
conv2d_4_45616:	?*
batch_normalization_4_45620:	?*
batch_normalization_4_45622:	?*
batch_normalization_4_45624:	?*
batch_normalization_4_45626:	?
dense_45636:	?$
dense_45638:)
batch_normalization_5_45641:)
batch_normalization_5_45643:)
batch_normalization_5_45645:)
batch_normalization_5_45647:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_45550conv2d_45552*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44803?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44333?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_45556batch_normalization_45558batch_normalization_45560batch_normalization_45562*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44358?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????bb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_44824?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_45566conv2d_1_45568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44836?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44409?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_45572batch_normalization_1_45574batch_normalization_1_45576batch_normalization_1_45578*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44434?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_44857?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_45582conv2d_2_45584*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44869?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_44485?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_45588batch_normalization_2_45590batch_normalization_2_45592batch_normalization_2_45594*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44510?
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_44890?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_45598conv2d_3_45600*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44902?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_44561?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_45604batch_normalization_3_45606batch_normalization_3_45608batch_normalization_3_45610*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44586?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_44923?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_45614conv2d_4_45616*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44935?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_44637?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_45620batch_normalization_4_45622batch_normalization_4_45624batch_normalization_4_45626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44662?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_44956g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:???????????
dropout/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44965i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
transpose_1	Transpose dropout/PartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:???????????
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44975?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_45636dense_45638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44988?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_45641batch_normalization_5_45643batch_normalization_5_45645batch_normalization_5_45647*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44728?
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_45008r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48498

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
"__inference__update_step_xla_47380
gradient
variable:	?$!
readvariableop_resource:	 #
readvariableop_1_resource: 0
sub_2_readvariableop_resource:	?$0
sub_3_readvariableop_resource:	?$??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: s
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
:	?$*
dtype0^
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	?$L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=S
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:	?$?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0D
SquareSquaregradient*
T0*
_output_shapes
:	?$s
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:	?$*
dtype0`
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	?$L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:S
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:	?$?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:	?$*
dtype0]
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:	?$?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:	?$*
dtype0W
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?$L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3V
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:	?$T
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:	?$f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:	?$: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:I E

_output_shapes
:	?$
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44902

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44510

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_3_layer_call_fn_48462

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44617?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_5_layer_call_fn_48693

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44775o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_2_layer_call_fn_48361

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44541?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_44857

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_48113

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44803y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_46863
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
? 
?
"__inference__update_step_xla_47239
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
K
/__inference_max_pooling2d_4_layer_call_fn_48532

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_44637?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_3_layer_call_fn_48503

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_44923i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_44923

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_47333
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?"
?
"__inference__update_step_xla_47192
gradient$
variable:??!
readvariableop_resource:	 #
readvariableop_1_resource: 9
sub_2_readvariableop_resource:??9
sub_3_readvariableop_resource:????AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: |
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*(
_output_shapes
:??*
dtype0g
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=\
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*(
_output_shapes
:???
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0M
SquareSquaregradient*
T0*(
_output_shapes
:??|
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*(
_output_shapes
:??*
dtype0i
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:\
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*(
_output_shapes
:???
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*(
_output_shapes
:??*
dtype0f
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*(
_output_shapes
:???
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*(
_output_shapes
:??*
dtype0`
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3_
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*(
_output_shapes
:??]
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*(
_output_shapes
:??f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:R N
(
_output_shapes
:??
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_48609

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
Ь
?-
__inference__traced_save_49210
file_prefix&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop0
,savev2_htc_conv2d_kernel_read_readvariableop.
*savev2_htc_conv2d_bias_read_readvariableop<
8savev2_htc_batch_normalization_gamma_read_readvariableop;
7savev2_htc_batch_normalization_beta_read_readvariableopB
>savev2_htc_batch_normalization_moving_mean_read_readvariableopF
Bsavev2_htc_batch_normalization_moving_variance_read_readvariableop2
.savev2_htc_conv2d_1_kernel_read_readvariableop0
,savev2_htc_conv2d_1_bias_read_readvariableop>
:savev2_htc_batch_normalization_1_gamma_read_readvariableop=
9savev2_htc_batch_normalization_1_beta_read_readvariableopD
@savev2_htc_batch_normalization_1_moving_mean_read_readvariableopH
Dsavev2_htc_batch_normalization_1_moving_variance_read_readvariableop2
.savev2_htc_conv2d_2_kernel_read_readvariableop0
,savev2_htc_conv2d_2_bias_read_readvariableop>
:savev2_htc_batch_normalization_2_gamma_read_readvariableop=
9savev2_htc_batch_normalization_2_beta_read_readvariableopD
@savev2_htc_batch_normalization_2_moving_mean_read_readvariableopH
Dsavev2_htc_batch_normalization_2_moving_variance_read_readvariableop2
.savev2_htc_conv2d_3_kernel_read_readvariableop0
,savev2_htc_conv2d_3_bias_read_readvariableop>
:savev2_htc_batch_normalization_3_gamma_read_readvariableop=
9savev2_htc_batch_normalization_3_beta_read_readvariableopD
@savev2_htc_batch_normalization_3_moving_mean_read_readvariableopH
Dsavev2_htc_batch_normalization_3_moving_variance_read_readvariableop2
.savev2_htc_conv2d_4_kernel_read_readvariableop0
,savev2_htc_conv2d_4_bias_read_readvariableop>
:savev2_htc_batch_normalization_4_gamma_read_readvariableop=
9savev2_htc_batch_normalization_4_beta_read_readvariableopD
@savev2_htc_batch_normalization_4_moving_mean_read_readvariableopH
Dsavev2_htc_batch_normalization_4_moving_variance_read_readvariableop/
+savev2_htc_dense_kernel_read_readvariableop-
)savev2_htc_dense_bias_read_readvariableop>
:savev2_htc_batch_normalization_5_gamma_read_readvariableop=
9savev2_htc_batch_normalization_5_beta_read_readvariableopD
@savev2_htc_batch_normalization_5_moving_mean_read_readvariableopH
Dsavev2_htc_batch_normalization_5_moving_variance_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop7
3savev2_adam_m_htc_conv2d_kernel_read_readvariableop7
3savev2_adam_v_htc_conv2d_kernel_read_readvariableop5
1savev2_adam_m_htc_conv2d_bias_read_readvariableop5
1savev2_adam_v_htc_conv2d_bias_read_readvariableopC
?savev2_adam_m_htc_batch_normalization_gamma_read_readvariableopC
?savev2_adam_v_htc_batch_normalization_gamma_read_readvariableopB
>savev2_adam_m_htc_batch_normalization_beta_read_readvariableopB
>savev2_adam_v_htc_batch_normalization_beta_read_readvariableop9
5savev2_adam_m_htc_conv2d_1_kernel_read_readvariableop9
5savev2_adam_v_htc_conv2d_1_kernel_read_readvariableop7
3savev2_adam_m_htc_conv2d_1_bias_read_readvariableop7
3savev2_adam_v_htc_conv2d_1_bias_read_readvariableopE
Asavev2_adam_m_htc_batch_normalization_1_gamma_read_readvariableopE
Asavev2_adam_v_htc_batch_normalization_1_gamma_read_readvariableopD
@savev2_adam_m_htc_batch_normalization_1_beta_read_readvariableopD
@savev2_adam_v_htc_batch_normalization_1_beta_read_readvariableop9
5savev2_adam_m_htc_conv2d_2_kernel_read_readvariableop9
5savev2_adam_v_htc_conv2d_2_kernel_read_readvariableop7
3savev2_adam_m_htc_conv2d_2_bias_read_readvariableop7
3savev2_adam_v_htc_conv2d_2_bias_read_readvariableopE
Asavev2_adam_m_htc_batch_normalization_2_gamma_read_readvariableopE
Asavev2_adam_v_htc_batch_normalization_2_gamma_read_readvariableopD
@savev2_adam_m_htc_batch_normalization_2_beta_read_readvariableopD
@savev2_adam_v_htc_batch_normalization_2_beta_read_readvariableop9
5savev2_adam_m_htc_conv2d_3_kernel_read_readvariableop9
5savev2_adam_v_htc_conv2d_3_kernel_read_readvariableop7
3savev2_adam_m_htc_conv2d_3_bias_read_readvariableop7
3savev2_adam_v_htc_conv2d_3_bias_read_readvariableopE
Asavev2_adam_m_htc_batch_normalization_3_gamma_read_readvariableopE
Asavev2_adam_v_htc_batch_normalization_3_gamma_read_readvariableopD
@savev2_adam_m_htc_batch_normalization_3_beta_read_readvariableopD
@savev2_adam_v_htc_batch_normalization_3_beta_read_readvariableop9
5savev2_adam_m_htc_conv2d_4_kernel_read_readvariableop9
5savev2_adam_v_htc_conv2d_4_kernel_read_readvariableop7
3savev2_adam_m_htc_conv2d_4_bias_read_readvariableop7
3savev2_adam_v_htc_conv2d_4_bias_read_readvariableopE
Asavev2_adam_m_htc_batch_normalization_4_gamma_read_readvariableopE
Asavev2_adam_v_htc_batch_normalization_4_gamma_read_readvariableopD
@savev2_adam_m_htc_batch_normalization_4_beta_read_readvariableopD
@savev2_adam_v_htc_batch_normalization_4_beta_read_readvariableop6
2savev2_adam_m_htc_dense_kernel_read_readvariableop6
2savev2_adam_v_htc_dense_kernel_read_readvariableop4
0savev2_adam_m_htc_dense_bias_read_readvariableop4
0savev2_adam_v_htc_dense_bias_read_readvariableopE
Asavev2_adam_m_htc_batch_normalization_5_gamma_read_readvariableopE
Asavev2_adam_v_htc_batch_normalization_5_gamma_read_readvariableopD
@savev2_adam_m_htc_batch_normalization_5_beta_read_readvariableopD
@savev2_adam_v_htc_batch_normalization_5_beta_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*?"
value?"B?"_B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*?
value?B?_B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop,savev2_htc_conv2d_kernel_read_readvariableop*savev2_htc_conv2d_bias_read_readvariableop8savev2_htc_batch_normalization_gamma_read_readvariableop7savev2_htc_batch_normalization_beta_read_readvariableop>savev2_htc_batch_normalization_moving_mean_read_readvariableopBsavev2_htc_batch_normalization_moving_variance_read_readvariableop.savev2_htc_conv2d_1_kernel_read_readvariableop,savev2_htc_conv2d_1_bias_read_readvariableop:savev2_htc_batch_normalization_1_gamma_read_readvariableop9savev2_htc_batch_normalization_1_beta_read_readvariableop@savev2_htc_batch_normalization_1_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_1_moving_variance_read_readvariableop.savev2_htc_conv2d_2_kernel_read_readvariableop,savev2_htc_conv2d_2_bias_read_readvariableop:savev2_htc_batch_normalization_2_gamma_read_readvariableop9savev2_htc_batch_normalization_2_beta_read_readvariableop@savev2_htc_batch_normalization_2_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_2_moving_variance_read_readvariableop.savev2_htc_conv2d_3_kernel_read_readvariableop,savev2_htc_conv2d_3_bias_read_readvariableop:savev2_htc_batch_normalization_3_gamma_read_readvariableop9savev2_htc_batch_normalization_3_beta_read_readvariableop@savev2_htc_batch_normalization_3_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_3_moving_variance_read_readvariableop.savev2_htc_conv2d_4_kernel_read_readvariableop,savev2_htc_conv2d_4_bias_read_readvariableop:savev2_htc_batch_normalization_4_gamma_read_readvariableop9savev2_htc_batch_normalization_4_beta_read_readvariableop@savev2_htc_batch_normalization_4_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_4_moving_variance_read_readvariableop+savev2_htc_dense_kernel_read_readvariableop)savev2_htc_dense_bias_read_readvariableop:savev2_htc_batch_normalization_5_gamma_read_readvariableop9savev2_htc_batch_normalization_5_beta_read_readvariableop@savev2_htc_batch_normalization_5_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_5_moving_variance_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop3savev2_adam_m_htc_conv2d_kernel_read_readvariableop3savev2_adam_v_htc_conv2d_kernel_read_readvariableop1savev2_adam_m_htc_conv2d_bias_read_readvariableop1savev2_adam_v_htc_conv2d_bias_read_readvariableop?savev2_adam_m_htc_batch_normalization_gamma_read_readvariableop?savev2_adam_v_htc_batch_normalization_gamma_read_readvariableop>savev2_adam_m_htc_batch_normalization_beta_read_readvariableop>savev2_adam_v_htc_batch_normalization_beta_read_readvariableop5savev2_adam_m_htc_conv2d_1_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_1_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_1_bias_read_readvariableop3savev2_adam_v_htc_conv2d_1_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_1_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_1_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_1_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_1_beta_read_readvariableop5savev2_adam_m_htc_conv2d_2_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_2_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_2_bias_read_readvariableop3savev2_adam_v_htc_conv2d_2_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_2_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_2_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_2_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_2_beta_read_readvariableop5savev2_adam_m_htc_conv2d_3_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_3_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_3_bias_read_readvariableop3savev2_adam_v_htc_conv2d_3_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_3_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_3_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_3_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_3_beta_read_readvariableop5savev2_adam_m_htc_conv2d_4_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_4_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_4_bias_read_readvariableop3savev2_adam_v_htc_conv2d_4_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_4_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_4_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_4_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_4_beta_read_readvariableop2savev2_adam_m_htc_dense_kernel_read_readvariableop2savev2_adam_v_htc_dense_kernel_read_readvariableop0savev2_adam_m_htc_dense_bias_read_readvariableop0savev2_adam_v_htc_dense_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_5_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_5_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_5_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_5_beta_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *m
dtypesc
a2_	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : @:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?:	?$:::::: : : : : : : : : : : @: @:@:@:@:@:@:@:@?:@?:?:?:?:?:?:?:??:??:?:?:?:?:?:?:??:??:?:?:?:?:?:?:	?$:	?$::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,	(
&
_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:! 

_output_shapes	
:?:.!*
(
_output_shapes
:??:!"

_output_shapes	
:?:!#

_output_shapes	
:?:!$

_output_shapes	
:?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:%'!

_output_shapes
:	?$: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::-

_output_shapes
: :.

_output_shapes
: :,/(
&
_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :,7(
&
_output_shapes
: @:,8(
&
_output_shapes
: @: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@: <

_output_shapes
:@: =

_output_shapes
:@: >

_output_shapes
:@:-?)
'
_output_shapes
:@?:-@)
'
_output_shapes
:@?:!A

_output_shapes	
:?:!B

_output_shapes	
:?:!C

_output_shapes	
:?:!D

_output_shapes	
:?:!E

_output_shapes	
:?:!F

_output_shapes	
:?:.G*
(
_output_shapes
:??:.H*
(
_output_shapes
:??:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:!L

_output_shapes	
:?:!M

_output_shapes	
:?:!N

_output_shapes	
:?:.O*
(
_output_shapes
:??:.P*
(
_output_shapes
:??:!Q

_output_shapes	
:?:!R

_output_shapes	
:?:!S

_output_shapes	
:?:!T

_output_shapes	
:?:!U

_output_shapes	
:?:!V

_output_shapes	
:?:%W!

_output_shapes
:	?$:%X!

_output_shapes
:	?$: Y

_output_shapes
:: Z

_output_shapes
:: [

_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
:: ^

_output_shapes
::_

_output_shapes
: 
?
?
%__inference_dense_layer_call_fn_48656

inputs
unknown:	?$
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44988o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48713

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
"__inference__update_step_xla_46628
gradient"
variable: @!
readvariableop_resource:	 #
readvariableop_1_resource: 7
sub_2_readvariableop_resource: @7
sub_3_readvariableop_resource: @??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: z
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*&
_output_shapes
: @*
dtype0e
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*&
_output_shapes
: @L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=Z
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*&
_output_shapes
: @?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0K
SquareSquaregradient*
T0*&
_output_shapes
: @z
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*&
_output_shapes
: @*
dtype0g
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*&
_output_shapes
: @L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:Z
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*&
_output_shapes
: @?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*&
_output_shapes
: @*
dtype0d
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*&
_output_shapes
: @?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*&
_output_shapes
: @*
dtype0^
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*&
_output_shapes
: @L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3]
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*&
_output_shapes
: @[
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*&
_output_shapes
: @f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*/
_input_shapes
: @: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:P L
&
_output_shapes
: @
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
(__inference_conv2d_4_layer_call_fn_48517

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44935x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44728

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_46581
gradient
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource: +
sub_3_readvariableop_resource: ??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: n
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes
: *
dtype0Y
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0?
SquareSquaregradient*
T0*
_output_shapes
: n
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype0[
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
: ?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0R
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
: L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3Q
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
: O
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
: f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48480

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44389

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
? 
?
"__inference__update_step_xla_47098
gradient
variable:	?!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	?,
sub_3_readvariableop_resource:	???AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: o
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*
_output_shapes	
:?*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:?o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:?*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:??
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:?*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:??
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:?*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:?P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:?f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:?: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48527

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
"__inference__update_step_xla_46438
gradient"
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: 7
sub_2_readvariableop_resource: 7
sub_3_readvariableop_resource: ??AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: z
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*&
_output_shapes
: *
dtype0e
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*&
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=Z
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*&
_output_shapes
: ?
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0K
SquareSquaregradient*
T0*&
_output_shapes
: z
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*&
_output_shapes
: *
dtype0g
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*&
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:Z
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*&
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*&
_output_shapes
: *
dtype0d
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*&
_output_shapes
: ?
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*&
_output_shapes
: *
dtype0^
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*&
_output_shapes
: L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3]
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*&
_output_shapes
: [
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*&
_output_shapes
: f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?"
?
"__inference__update_step_xla_47004
gradient$
variable:??!
readvariableop_resource:	 #
readvariableop_1_resource: 9
sub_2_readvariableop_resource:??9
sub_3_readvariableop_resource:????AssignAddVariableOp?AssignAddVariableOp_1?AssignSubVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?Sqrt_1/ReadVariableOp?sub_2/ReadVariableOp?sub_3/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: E
CastCastadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?H
PowPowCast_1/x:output:0Cast:y:0*
T0*
_output_shapes
: M
Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w??J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??F
subSubsub/x:output:0	Pow_1:z:0*
T0*
_output_shapes
: 6
SqrtSqrtsub:z:0*
T0*
_output_shapes
: b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0O
mulMulReadVariableOp_1:value:0Sqrt:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
sub_1Subsub_1/x:output:0Pow:z:0*
T0*
_output_shapes
: G
truedivRealDivmul:z:0	sub_1:z:0*
T0*
_output_shapes
: |
sub_2/ReadVariableOpReadVariableOpsub_2_readvariableop_resource*(
_output_shapes
:??*
dtype0g
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=\
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*(
_output_shapes
:???
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0M
SquareSquaregradient*
T0*(
_output_shapes
:??|
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*(
_output_shapes
:??*
dtype0i
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:\
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*(
_output_shapes
:???
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0?
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*(
_output_shapes
:??*
dtype0f
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*(
_output_shapes
:???
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*(
_output_shapes
:??*
dtype0`
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3_
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*(
_output_shapes
:??]
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*(
_output_shapes
:??f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??: : : : : *
	_noinline(2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12*
AssignSubVariableOpAssignSubVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22.
Sqrt_1/ReadVariableOpSqrt_1/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp:R N
(
_output_shapes
:??
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?	
?
5__inference_batch_normalization_1_layer_call_fn_48260

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44465?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:
"__inference_internal_grad_fn_49096CustomGradient-46326"?
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	
train_loss

train_accuracy
	test_loss
test_accuracy
	conv1
	pool1

batch1
relu_activ1
	conv2
	pool2

batch2
relu_activ2
	conv3
	pool3

batch3
relu_activ3
	conv4
	pool4

batch4
relu_activ4
	conv5
	pool5

batch5
 relu_activ5
!dropout1
"flatten
	#d_fin
$	batch_fin
%
soft_activ
&	test_step
'
train_step
(
signatures"
_tf_keras_model
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29
G30
H31
I32
J33
K34
L35
M36
N37
O38
P39
Q40
R41
S42
T43"
trackable_list_wrapper
?
10
21
32
43
74
85
96
:7
=8
>9
?10
@11
C12
D13
E14
F15
I16
J17
K18
L19
O20
P21
Q22
R23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Ztrace_0
[trace_1
\trace_2
]trace_32?
#__inference_htc_layer_call_fn_45086
#__inference_htc_layer_call_fn_47716
#__inference_htc_layer_call_fn_47793
#__inference_htc_layer_call_fn_45547?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 zZtrace_0z[trace_1z\trace_2z]trace_3
?
^trace_0
_trace_1
`trace_2
atrace_32?
>__inference_htc_layer_call_and_return_conditional_losses_47938
>__inference_htc_layer_call_and_return_conditional_losses_48104
>__inference_htc_layer_call_and_return_conditional_losses_45652
>__inference_htc_layer_call_and_return_conditional_losses_45757?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z^trace_0z_trace_1z`trace_2zatrace_3
?B?
 __inference__wrapped_model_44324input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
b
_variables
c_iterations
d_learning_rate
e_index_dict
f
_momentums
g_velocities
h_update_step_xla"
experimentalOptimizer
N
i	variables
j	keras_api
	)total
	*count"
_tf_keras_metric
^
k	variables
l	keras_api
	+total
	,count
m
_fn_kwargs"
_tf_keras_metric
N
n	variables
o	keras_api
	-total
	.count"
_tf_keras_metric
^
p	variables
q	keras_api
	/total
	0count
r
_fn_kwargs"
_tf_keras_metric
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

1kernel
2bias
 y_jit_compiled_convolution_op"
_tf_keras_layer
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	3gamma
4beta
5moving_mean
6moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

7kernel
8bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	9gamma
:beta
;moving_mean
<moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

=kernel
>bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	?gamma
@beta
Amoving_mean
Bmoving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Ckernel
Dbias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Ikernel
Jbias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Okernel
Pbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_02?
__inference_test_step_45950?
???
FullArgSpec'
args?
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
__inference_train_step_47560?
???
FullArgSpec'
args?
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
-
?serving_default"
signature_map
:  (2total
:  (2count
:  (2total
:  (2count
:  (2total
:  (2count
:  (2total
:  (2count
+:) 2htc/conv2d/kernel
: 2htc/conv2d/bias
+:) 2htc/batch_normalization/gamma
*:( 2htc/batch_normalization/beta
3:1  (2#htc/batch_normalization/moving_mean
7:5  (2'htc/batch_normalization/moving_variance
-:+ @2htc/conv2d_1/kernel
:@2htc/conv2d_1/bias
-:+@2htc/batch_normalization_1/gamma
,:*@2htc/batch_normalization_1/beta
5:3@ (2%htc/batch_normalization_1/moving_mean
9:7@ (2)htc/batch_normalization_1/moving_variance
.:,@?2htc/conv2d_2/kernel
 :?2htc/conv2d_2/bias
.:,?2htc/batch_normalization_2/gamma
-:+?2htc/batch_normalization_2/beta
6:4? (2%htc/batch_normalization_2/moving_mean
::8? (2)htc/batch_normalization_2/moving_variance
/:-??2htc/conv2d_3/kernel
 :?2htc/conv2d_3/bias
.:,?2htc/batch_normalization_3/gamma
-:+?2htc/batch_normalization_3/beta
6:4? (2%htc/batch_normalization_3/moving_mean
::8? (2)htc/batch_normalization_3/moving_variance
/:-??2htc/conv2d_4/kernel
 :?2htc/conv2d_4/bias
.:,?2htc/batch_normalization_4/gamma
-:+?2htc/batch_normalization_4/beta
6:4? (2%htc/batch_normalization_4/moving_mean
::8? (2)htc/batch_normalization_4/moving_variance
#:!	?$2htc/dense/kernel
:2htc/dense/bias
-:+2htc/batch_normalization_5/gamma
,:*2htc/batch_normalization_5/beta
5:3 (2%htc/batch_normalization_5/moving_mean
9:7 (2)htc/batch_normalization_5/moving_variance
?
)0
*1
+2
,3
-4
.5
/6
07
58
69
;10
<11
A12
B13
G14
H15
M16
N17
S18
T19"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23
%24"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
f
	
train_loss

train_accuracy
	test_loss
test_accuracy"
trackable_dict_wrapper
?B?
#__inference_htc_layer_call_fn_45086input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
#__inference_htc_layer_call_fn_47716x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
#__inference_htc_layer_call_fn_47793x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
#__inference_htc_layer_call_fn_45547input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
>__inference_htc_layer_call_and_return_conditional_losses_47938x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
>__inference_htc_layer_call_and_return_conditional_losses_48104x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
>__inference_htc_layer_call_and_return_conditional_losses_45652input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
>__inference_htc_layer_call_and_return_conditional_losses_45757input_1"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?
c0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
?2??
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
)0
*1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
 "
trackable_dict_wrapper
.
-0
.1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
.
/0
01"
trackable_list_wrapper
-
p	variables"
_generic_user_object
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
&__inference_conv2d_layer_call_fn_48113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
A__inference_conv2d_layer_call_and_return_conditional_losses_48123?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_max_pooling2d_layer_call_fn_48128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_48133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
3__inference_batch_normalization_layer_call_fn_48146
3__inference_batch_normalization_layer_call_fn_48159?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48177
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48195?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_re_lu_layer_call_fn_48200?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_re_lu_layer_call_and_return_conditional_losses_48205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_1_layer_call_fn_48214?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48224?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_1_layer_call_fn_48229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48234?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_1_layer_call_fn_48247
5__inference_batch_normalization_1_layer_call_fn_48260?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48278
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48296?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_re_lu_1_layer_call_fn_48301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_re_lu_1_layer_call_and_return_conditional_losses_48306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_2_layer_call_fn_48315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_2_layer_call_fn_48330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_48335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_2_layer_call_fn_48348
5__inference_batch_normalization_2_layer_call_fn_48361?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48379
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48397?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_re_lu_2_layer_call_fn_48402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_re_lu_2_layer_call_and_return_conditional_losses_48407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_3_layer_call_fn_48416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_3_layer_call_fn_48431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_48436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_3_layer_call_fn_48449
5__inference_batch_normalization_3_layer_call_fn_48462?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48480
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48498?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_re_lu_3_layer_call_fn_48503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_re_lu_3_layer_call_and_return_conditional_losses_48508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_4_layer_call_fn_48517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48527?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_4_layer_call_fn_48532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_48537?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_4_layer_call_fn_48550
5__inference_batch_normalization_4_layer_call_fn_48563?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48581
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48599?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_re_lu_4_layer_call_fn_48604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_re_lu_4_layer_call_and_return_conditional_losses_48609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
'__inference_dropout_layer_call_fn_48614
'__inference_dropout_layer_call_fn_48619?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_48624
B__inference_dropout_layer_call_and_return_conditional_losses_48636?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_flatten_layer_call_fn_48641?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_flatten_layer_call_and_return_conditional_losses_48647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_dense_layer_call_fn_48656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_dense_layer_call_and_return_conditional_losses_48667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_5_layer_call_fn_48680
5__inference_batch_normalization_5_layer_call_fn_48693?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48713
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48747?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_activation_layer_call_fn_48752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_activation_layer_call_and_return_conditional_losses_48757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?B?
__inference_test_step_45950imageslabels"?
???
FullArgSpec'
args?
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_train_step_47560imageslabels"?
???
FullArgSpec'
args?
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_47639input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0:. 2Adam/m/htc/conv2d/kernel
0:. 2Adam/v/htc/conv2d/kernel
":  2Adam/m/htc/conv2d/bias
":  2Adam/v/htc/conv2d/bias
0:. 2$Adam/m/htc/batch_normalization/gamma
0:. 2$Adam/v/htc/batch_normalization/gamma
/:- 2#Adam/m/htc/batch_normalization/beta
/:- 2#Adam/v/htc/batch_normalization/beta
2:0 @2Adam/m/htc/conv2d_1/kernel
2:0 @2Adam/v/htc/conv2d_1/kernel
$:"@2Adam/m/htc/conv2d_1/bias
$:"@2Adam/v/htc/conv2d_1/bias
2:0@2&Adam/m/htc/batch_normalization_1/gamma
2:0@2&Adam/v/htc/batch_normalization_1/gamma
1:/@2%Adam/m/htc/batch_normalization_1/beta
1:/@2%Adam/v/htc/batch_normalization_1/beta
3:1@?2Adam/m/htc/conv2d_2/kernel
3:1@?2Adam/v/htc/conv2d_2/kernel
%:#?2Adam/m/htc/conv2d_2/bias
%:#?2Adam/v/htc/conv2d_2/bias
3:1?2&Adam/m/htc/batch_normalization_2/gamma
3:1?2&Adam/v/htc/batch_normalization_2/gamma
2:0?2%Adam/m/htc/batch_normalization_2/beta
2:0?2%Adam/v/htc/batch_normalization_2/beta
4:2??2Adam/m/htc/conv2d_3/kernel
4:2??2Adam/v/htc/conv2d_3/kernel
%:#?2Adam/m/htc/conv2d_3/bias
%:#?2Adam/v/htc/conv2d_3/bias
3:1?2&Adam/m/htc/batch_normalization_3/gamma
3:1?2&Adam/v/htc/batch_normalization_3/gamma
2:0?2%Adam/m/htc/batch_normalization_3/beta
2:0?2%Adam/v/htc/batch_normalization_3/beta
4:2??2Adam/m/htc/conv2d_4/kernel
4:2??2Adam/v/htc/conv2d_4/kernel
%:#?2Adam/m/htc/conv2d_4/bias
%:#?2Adam/v/htc/conv2d_4/bias
3:1?2&Adam/m/htc/batch_normalization_4/gamma
3:1?2&Adam/v/htc/batch_normalization_4/gamma
2:0?2%Adam/m/htc/batch_normalization_4/beta
2:0?2%Adam/v/htc/batch_normalization_4/beta
(:&	?$2Adam/m/htc/dense/kernel
(:&	?$2Adam/v/htc/dense/kernel
!:2Adam/m/htc/dense/bias
!:2Adam/v/htc/dense/bias
2:02&Adam/m/htc/batch_normalization_5/gamma
2:02&Adam/v/htc/batch_normalization_5/gamma
1:/2%Adam/m/htc/batch_normalization_5/beta
1:/2%Adam/v/htc/batch_normalization_5/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_conv2d_layer_call_fn_48113inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_conv2d_layer_call_and_return_conditional_losses_48123inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_max_pooling2d_layer_call_fn_48128inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_48133inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_batch_normalization_layer_call_fn_48146inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
3__inference_batch_normalization_layer_call_fn_48159inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48177inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48195inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_re_lu_layer_call_fn_48200inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_re_lu_layer_call_and_return_conditional_losses_48205inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_1_layer_call_fn_48214inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48224inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_max_pooling2d_1_layer_call_fn_48229inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48234inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_1_layer_call_fn_48247inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_batch_normalization_1_layer_call_fn_48260inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48278inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48296inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_re_lu_1_layer_call_fn_48301inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_re_lu_1_layer_call_and_return_conditional_losses_48306inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_2_layer_call_fn_48315inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48325inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_max_pooling2d_2_layer_call_fn_48330inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_48335inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_2_layer_call_fn_48348inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_batch_normalization_2_layer_call_fn_48361inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48379inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48397inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_re_lu_2_layer_call_fn_48402inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_re_lu_2_layer_call_and_return_conditional_losses_48407inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_3_layer_call_fn_48416inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48426inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_max_pooling2d_3_layer_call_fn_48431inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_48436inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_3_layer_call_fn_48449inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_batch_normalization_3_layer_call_fn_48462inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48480inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48498inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_re_lu_3_layer_call_fn_48503inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_re_lu_3_layer_call_and_return_conditional_losses_48508inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_4_layer_call_fn_48517inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48527inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_max_pooling2d_4_layer_call_fn_48532inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_48537inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_4_layer_call_fn_48550inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_batch_normalization_4_layer_call_fn_48563inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48581inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48599inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_re_lu_4_layer_call_fn_48604inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_re_lu_4_layer_call_and_return_conditional_losses_48609inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dropout_layer_call_fn_48614inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_dropout_layer_call_fn_48619inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_48624inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_48636inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_flatten_layer_call_fn_48641inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_flatten_layer_call_and_return_conditional_losses_48647inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_dense_layer_call_fn_48656inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_48667inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_5_layer_call_fn_48680inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
5__inference_batch_normalization_5_layer_call_fn_48693inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48713inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48747inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_activation_layer_call_fn_48752inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_activation_layer_call_and_return_conditional_losses_48757inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_44324?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQ:?7
0?-
+?(
input_1???????????
? "3?0
.
output_1"?
output_1??????????
E__inference_activation_layer_call_and_return_conditional_losses_48757_/?,
%?"
 ?
inputs?????????
? ",?)
"?
tensor_0?????????
? ?
*__inference_activation_layer_call_fn_48752T/?,
%?"
 ?
inputs?????????
? "!?
unknown??????????
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48278?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "F?C
<?9
tensor_0+???????????????????????????@
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48296?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p
? "F?C
<?9
tensor_0+???????????????????????????@
? ?
5__inference_batch_normalization_1_layer_call_fn_48247?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p 
? ";?8
unknown+???????????????????????????@?
5__inference_batch_normalization_1_layer_call_fn_48260?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p
? ";?8
unknown+???????????????????????????@?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48379??@ABN?K
D?A
;?8
inputs,????????????????????????????
p 
? "G?D
=?:
tensor_0,????????????????????????????
? ?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48397??@ABN?K
D?A
;?8
inputs,????????????????????????????
p
? "G?D
=?:
tensor_0,????????????????????????????
? ?
5__inference_batch_normalization_2_layer_call_fn_48348??@ABN?K
D?A
;?8
inputs,????????????????????????????
p 
? "<?9
unknown,?????????????????????????????
5__inference_batch_normalization_2_layer_call_fn_48361??@ABN?K
D?A
;?8
inputs,????????????????????????????
p
? "<?9
unknown,?????????????????????????????
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48480?EFGHN?K
D?A
;?8
inputs,????????????????????????????
p 
? "G?D
=?:
tensor_0,????????????????????????????
? ?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48498?EFGHN?K
D?A
;?8
inputs,????????????????????????????
p
? "G?D
=?:
tensor_0,????????????????????????????
? ?
5__inference_batch_normalization_3_layer_call_fn_48449?EFGHN?K
D?A
;?8
inputs,????????????????????????????
p 
? "<?9
unknown,?????????????????????????????
5__inference_batch_normalization_3_layer_call_fn_48462?EFGHN?K
D?A
;?8
inputs,????????????????????????????
p
? "<?9
unknown,?????????????????????????????
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48581?KLMNN?K
D?A
;?8
inputs,????????????????????????????
p 
? "G?D
=?:
tensor_0,????????????????????????????
? ?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48599?KLMNN?K
D?A
;?8
inputs,????????????????????????????
p
? "G?D
=?:
tensor_0,????????????????????????????
? ?
5__inference_batch_normalization_4_layer_call_fn_48550?KLMNN?K
D?A
;?8
inputs,????????????????????????????
p 
? "<?9
unknown,?????????????????????????????
5__inference_batch_normalization_4_layer_call_fn_48563?KLMNN?K
D?A
;?8
inputs,????????????????????????????
p
? "<?9
unknown,?????????????????????????????
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48713iSTRQ3?0
)?&
 ?
inputs?????????
p 
? ",?)
"?
tensor_0?????????
? ?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48747iSTRQ3?0
)?&
 ?
inputs?????????
p
? ",?)
"?
tensor_0?????????
? ?
5__inference_batch_normalization_5_layer_call_fn_48680^STRQ3?0
)?&
 ?
inputs?????????
p 
? "!?
unknown??????????
5__inference_batch_normalization_5_layer_call_fn_48693^STRQ3?0
)?&
 ?
inputs?????????
p
? "!?
unknown??????????
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48177?3456M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "F?C
<?9
tensor_0+??????????????????????????? 
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48195?3456M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "F?C
<?9
tensor_0+??????????????????????????? 
? ?
3__inference_batch_normalization_layer_call_fn_48146?3456M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? ";?8
unknown+??????????????????????????? ?
3__inference_batch_normalization_layer_call_fn_48159?3456M?J
C?@
:?7
inputs+??????????????????????????? 
p
? ";?8
unknown+??????????????????????????? ?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48224s787?4
-?*
(?%
inputs?????????bb 
? "4?1
*?'
tensor_0?????????^^@
? ?
(__inference_conv2d_1_layer_call_fn_48214h787?4
-?*
(?%
inputs?????????bb 
? ")?&
unknown?????????^^@?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48325t=>7?4
-?*
(?%
inputs?????????@
? "5?2
+?(
tensor_0??????????
? ?
(__inference_conv2d_2_layer_call_fn_48315i=>7?4
-?*
(?%
inputs?????????@
? "*?'
unknown???????????
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48426uCD8?5
.?+
)?&
inputs??????????
? "5?2
+?(
tensor_0??????????
? ?
(__inference_conv2d_3_layer_call_fn_48416jCD8?5
.?+
)?&
inputs??????????
? "*?'
unknown???????????
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48527uIJ8?5
.?+
)?&
inputs??????????
? "5?2
+?(
tensor_0??????????
? ?
(__inference_conv2d_4_layer_call_fn_48517jIJ8?5
.?+
)?&
inputs??????????
? "*?'
unknown???????????
A__inference_conv2d_layer_call_and_return_conditional_losses_48123w129?6
/?,
*?'
inputs???????????
? "6?3
,?)
tensor_0??????????? 
? ?
&__inference_conv2d_layer_call_fn_48113l129?6
/?,
*?'
inputs???????????
? "+?(
unknown??????????? ?
@__inference_dense_layer_call_and_return_conditional_losses_48667dOP0?-
&?#
!?
inputs??????????$
? ",?)
"?
tensor_0?????????
? ?
%__inference_dense_layer_call_fn_48656YOP0?-
&?#
!?
inputs??????????$
? "!?
unknown??????????
B__inference_dropout_layer_call_and_return_conditional_losses_48624u<?9
2?/
)?&
inputs??????????
p 
? "5?2
+?(
tensor_0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_48636u<?9
2?/
)?&
inputs??????????
p
? "5?2
+?(
tensor_0??????????
? ?
'__inference_dropout_layer_call_fn_48614j<?9
2?/
)?&
inputs??????????
p 
? "*?'
unknown???????????
'__inference_dropout_layer_call_fn_48619j<?9
2?/
)?&
inputs??????????
p
? "*?'
unknown???????????
B__inference_flatten_layer_call_and_return_conditional_losses_48647i8?5
.?+
)?&
inputs??????????
? "-?*
#? 
tensor_0??????????$
? ?
'__inference_flatten_layer_call_fn_48641^8?5
.?+
)?&
inputs??????????
? ""?
unknown??????????$?
>__inference_htc_layer_call_and_return_conditional_losses_45652?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ?G
0?-
+?(
input_1???????????
?

trainingp ",?)
"?
tensor_0?????????
? ?
>__inference_htc_layer_call_and_return_conditional_losses_45757?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ?G
0?-
+?(
input_1???????????
?

trainingp",?)
"?
tensor_0?????????
? ?
>__inference_htc_layer_call_and_return_conditional_losses_47938?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD?A
*?'
%?"
x???????????
?

trainingp ",?)
"?
tensor_0?????????
? ?
>__inference_htc_layer_call_and_return_conditional_losses_48104?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD?A
*?'
%?"
x???????????
?

trainingp",?)
"?
tensor_0?????????
? ?
#__inference_htc_layer_call_fn_45086?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ?G
0?-
+?(
input_1???????????
?

trainingp "!?
unknown??????????
#__inference_htc_layer_call_fn_45547?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ?G
0?-
+?(
input_1???????????
?

trainingp"!?
unknown??????????
#__inference_htc_layer_call_fn_47716?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD?A
*?'
%?"
x???????????
?

trainingp "!?
unknown??????????
#__inference_htc_layer_call_fn_47793?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD?A
*?'
%?"
x???????????
?

trainingp"!?
unknown??????????
"__inference_internal_grad_fn_49096????
???

 
'?$
result_grads_0 
?
result_grads_1 
?
result_grads_2 
?
result_grads_3 
'?$
result_grads_4 @
?
result_grads_5@
?
result_grads_6@
?
result_grads_7@
(?%
result_grads_8@?
?
result_grads_9?
?
result_grads_10?
?
result_grads_11?
*?'
result_grads_12??
?
result_grads_13?
?
result_grads_14?
?
result_grads_15?
*?'
result_grads_16??
?
result_grads_17?
?
result_grads_18?
?
result_grads_19?
!?
result_grads_20	?$
?
result_grads_21
?
result_grads_22
?
result_grads_23
(?%
result_grads_24 
?
result_grads_25 
?
result_grads_26 
?
result_grads_27 
(?%
result_grads_28 @
?
result_grads_29@
?
result_grads_30@
?
result_grads_31@
)?&
result_grads_32@?
?
result_grads_33?
?
result_grads_34?
?
result_grads_35?
*?'
result_grads_36??
?
result_grads_37?
?
result_grads_38?
?
result_grads_39?
*?'
result_grads_40??
?
result_grads_41?
?
result_grads_42?
?
result_grads_43?
!?
result_grads_44	?$
?
result_grads_45
?
result_grads_46
?
result_grads_47
? "???

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 
"?
	tensor_24 
?
	tensor_25 
?
	tensor_26 
?
	tensor_27 
"?
	tensor_28 @
?
	tensor_29@
?
	tensor_30@
?
	tensor_31@
#? 
	tensor_32@?
?
	tensor_33?
?
	tensor_34?
?
	tensor_35?
$?!
	tensor_36??
?
	tensor_37?
?
	tensor_38?
?
	tensor_39?
$?!
	tensor_40??
?
	tensor_41?
?
	tensor_42?
?
	tensor_43?
?
	tensor_44	?$
?
	tensor_45
?
	tensor_46
?
	tensor_47?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48234?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_48229?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_48335?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_48330?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_48436?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
/__inference_max_pooling2d_3_layer_call_fn_48431?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_48537?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_48532?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_48133?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_48128?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
B__inference_re_lu_1_layer_call_and_return_conditional_losses_48306o7?4
-?*
(?%
inputs?????????@
? "4?1
*?'
tensor_0?????????@
? ?
'__inference_re_lu_1_layer_call_fn_48301d7?4
-?*
(?%
inputs?????????@
? ")?&
unknown?????????@?
B__inference_re_lu_2_layer_call_and_return_conditional_losses_48407q8?5
.?+
)?&
inputs??????????
? "5?2
+?(
tensor_0??????????
? ?
'__inference_re_lu_2_layer_call_fn_48402f8?5
.?+
)?&
inputs??????????
? "*?'
unknown???????????
B__inference_re_lu_3_layer_call_and_return_conditional_losses_48508q8?5
.?+
)?&
inputs??????????
? "5?2
+?(
tensor_0??????????
? ?
'__inference_re_lu_3_layer_call_fn_48503f8?5
.?+
)?&
inputs??????????
? "*?'
unknown???????????
B__inference_re_lu_4_layer_call_and_return_conditional_losses_48609q8?5
.?+
)?&
inputs??????????
? "5?2
+?(
tensor_0??????????
? ?
'__inference_re_lu_4_layer_call_fn_48604f8?5
.?+
)?&
inputs??????????
? "*?'
unknown???????????
@__inference_re_lu_layer_call_and_return_conditional_losses_48205o7?4
-?*
(?%
inputs?????????bb 
? "4?1
*?'
tensor_0?????????bb 
? ?
%__inference_re_lu_layer_call_fn_48200d7?4
-?*
(?%
inputs?????????bb 
? ")?&
unknown?????????bb ?
#__inference_signature_wrapper_47639?$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQE?B
? 
;?8
6
input_1+?(
input_1???????????"3?0
.
output_1"?
output_1??????????
__inference_test_step_45950u(123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQ-./0E?B
;?8
!?
images ??
?
labels 
? "
 ?
__inference_train_step_47560??123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQcd????????????????????????????????????????????????)*+,E?B
;?8
!?
images ??
?
labels 
? "
 