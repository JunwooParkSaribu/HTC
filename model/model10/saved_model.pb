é,
õÄ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
û
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
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
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

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
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
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ÃÉ'
¢
%Adam/v/htc/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/htc/batch_normalization_5/beta

9Adam/v/htc/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_5/beta*
_output_shapes
:*
dtype0
¢
%Adam/m/htc/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/htc/batch_normalization_5/beta

9Adam/m/htc/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_5/beta*
_output_shapes
:*
dtype0
¤
&Adam/v/htc/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/htc/batch_normalization_5/gamma

:Adam/v/htc/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
¤
&Adam/m/htc/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/htc/batch_normalization_5/gamma

:Adam/m/htc/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_5/gamma*
_output_shapes
:*
dtype0

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

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

Adam/v/htc/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$*(
shared_nameAdam/v/htc/dense/kernel

+Adam/v/htc/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/dense/kernel*
_output_shapes
:	$*
dtype0

Adam/m/htc/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$*(
shared_nameAdam/m/htc/dense/kernel

+Adam/m/htc/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/dense/kernel*
_output_shapes
:	$*
dtype0
£
%Adam/v/htc/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/htc/batch_normalization_4/beta

9Adam/v/htc/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_4/beta*
_output_shapes	
:*
dtype0
£
%Adam/m/htc/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/htc/batch_normalization_4/beta

9Adam/m/htc/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_4/beta*
_output_shapes	
:*
dtype0
¥
&Adam/v/htc/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/htc/batch_normalization_4/gamma

:Adam/v/htc/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0
¥
&Adam/m/htc/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/htc/batch_normalization_4/gamma

:Adam/m/htc/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0

Adam/v/htc/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/htc/conv2d_4/bias

,Adam/v/htc/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_4/bias*
_output_shapes	
:*
dtype0

Adam/m/htc/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/htc/conv2d_4/bias

,Adam/m/htc/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_4/bias*
_output_shapes	
:*
dtype0

Adam/v/htc/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/htc/conv2d_4/kernel

.Adam/v/htc/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_4/kernel*(
_output_shapes
:*
dtype0

Adam/m/htc/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/htc/conv2d_4/kernel

.Adam/m/htc/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_4/kernel*(
_output_shapes
:*
dtype0
£
%Adam/v/htc/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/htc/batch_normalization_3/beta

9Adam/v/htc/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_3/beta*
_output_shapes	
:*
dtype0
£
%Adam/m/htc/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/htc/batch_normalization_3/beta

9Adam/m/htc/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_3/beta*
_output_shapes	
:*
dtype0
¥
&Adam/v/htc/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/htc/batch_normalization_3/gamma

:Adam/v/htc/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_3/gamma*
_output_shapes	
:*
dtype0
¥
&Adam/m/htc/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/htc/batch_normalization_3/gamma

:Adam/m/htc/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_3/gamma*
_output_shapes	
:*
dtype0

Adam/v/htc/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/htc/conv2d_3/bias

,Adam/v/htc/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_3/bias*
_output_shapes	
:*
dtype0

Adam/m/htc/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/htc/conv2d_3/bias

,Adam/m/htc/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_3/bias*
_output_shapes	
:*
dtype0

Adam/v/htc/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/htc/conv2d_3/kernel

.Adam/v/htc/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_3/kernel*(
_output_shapes
:*
dtype0

Adam/m/htc/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/htc/conv2d_3/kernel

.Adam/m/htc/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_3/kernel*(
_output_shapes
:*
dtype0
£
%Adam/v/htc/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/htc/batch_normalization_2/beta

9Adam/v/htc/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_2/beta*
_output_shapes	
:*
dtype0
£
%Adam/m/htc/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/htc/batch_normalization_2/beta

9Adam/m/htc/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_2/beta*
_output_shapes	
:*
dtype0
¥
&Adam/v/htc/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/htc/batch_normalization_2/gamma

:Adam/v/htc/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_2/gamma*
_output_shapes	
:*
dtype0
¥
&Adam/m/htc/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/htc/batch_normalization_2/gamma

:Adam/m/htc/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_2/gamma*
_output_shapes	
:*
dtype0

Adam/v/htc/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/htc/conv2d_2/bias

,Adam/v/htc/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_2/bias*
_output_shapes	
:*
dtype0

Adam/m/htc/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/htc/conv2d_2/bias

,Adam/m/htc/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_2/bias*
_output_shapes	
:*
dtype0

Adam/v/htc/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/v/htc/conv2d_2/kernel

.Adam/v/htc/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_2/kernel*'
_output_shapes
:@*
dtype0

Adam/m/htc/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/m/htc/conv2d_2/kernel

.Adam/m/htc/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_2/kernel*'
_output_shapes
:@*
dtype0
¢
%Adam/v/htc/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/v/htc/batch_normalization_1/beta

9Adam/v/htc/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp%Adam/v/htc/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
¢
%Adam/m/htc/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/m/htc/batch_normalization_1/beta

9Adam/m/htc/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp%Adam/m/htc/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
¤
&Adam/v/htc/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/v/htc/batch_normalization_1/gamma

:Adam/v/htc/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp&Adam/v/htc/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
¤
&Adam/m/htc/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/m/htc/batch_normalization_1/gamma

:Adam/m/htc/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp&Adam/m/htc/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0

Adam/v/htc/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/v/htc/conv2d_1/bias

,Adam/v/htc/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_1/bias*
_output_shapes
:@*
dtype0

Adam/m/htc/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/m/htc/conv2d_1/bias

,Adam/m/htc/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_1/bias*
_output_shapes
:@*
dtype0

Adam/v/htc/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/v/htc/conv2d_1/kernel

.Adam/v/htc/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d_1/kernel*&
_output_shapes
: @*
dtype0

Adam/m/htc/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/m/htc/conv2d_1/kernel

.Adam/m/htc/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/htc/conv2d_1/kernel*&
_output_shapes
: @*
dtype0

#Adam/v/htc/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/v/htc/batch_normalization/beta

7Adam/v/htc/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#Adam/v/htc/batch_normalization/beta*
_output_shapes
: *
dtype0

#Adam/m/htc/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/m/htc/batch_normalization/beta

7Adam/m/htc/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#Adam/m/htc/batch_normalization/beta*
_output_shapes
: *
dtype0
 
$Adam/v/htc/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/v/htc/batch_normalization/gamma

8Adam/v/htc/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$Adam/v/htc/batch_normalization/gamma*
_output_shapes
: *
dtype0
 
$Adam/m/htc/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/m/htc/batch_normalization/gamma

8Adam/m/htc/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$Adam/m/htc/batch_normalization/gamma*
_output_shapes
: *
dtype0

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

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

Adam/v/htc/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/v/htc/conv2d/kernel

,Adam/v/htc/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/htc/conv2d/kernel*&
_output_shapes
: *
dtype0

Adam/m/htc/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/m/htc/conv2d/kernel

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
ª
)htc/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)htc/batch_normalization_5/moving_variance
£
=htc/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
¢
%htc/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%htc/batch_normalization_5/moving_mean

9htc/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0

htc/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name htc/batch_normalization_5/beta

2htc/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_5/beta*
_output_shapes
:*
dtype0

htc/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!htc/batch_normalization_5/gamma

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
shape:	$*!
shared_namehtc/dense/kernel
v
$htc/dense/kernel/Read/ReadVariableOpReadVariableOphtc/dense/kernel*
_output_shapes
:	$*
dtype0
«
)htc/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)htc/batch_normalization_4/moving_variance
¤
=htc/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0
£
%htc/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%htc/batch_normalization_4/moving_mean

9htc/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0

htc/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name htc/batch_normalization_4/beta

2htc/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_4/beta*
_output_shapes	
:*
dtype0

htc/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!htc/batch_normalization_4/gamma

3htc/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0
{
htc/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namehtc/conv2d_4/bias
t
%htc/conv2d_4/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_4/bias*
_output_shapes	
:*
dtype0

htc/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehtc/conv2d_4/kernel

'htc/conv2d_4/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_4/kernel*(
_output_shapes
:*
dtype0
«
)htc/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)htc/batch_normalization_3/moving_variance
¤
=htc/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0
£
%htc/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%htc/batch_normalization_3/moving_mean

9htc/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0

htc/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name htc/batch_normalization_3/beta

2htc/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_3/beta*
_output_shapes	
:*
dtype0

htc/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!htc/batch_normalization_3/gamma

3htc/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_3/gamma*
_output_shapes	
:*
dtype0
{
htc/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namehtc/conv2d_3/bias
t
%htc/conv2d_3/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_3/bias*
_output_shapes	
:*
dtype0

htc/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehtc/conv2d_3/kernel

'htc/conv2d_3/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_3/kernel*(
_output_shapes
:*
dtype0
«
)htc/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)htc/batch_normalization_2/moving_variance
¤
=htc/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_2/moving_variance*
_output_shapes	
:*
dtype0
£
%htc/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%htc/batch_normalization_2/moving_mean

9htc/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_2/moving_mean*
_output_shapes	
:*
dtype0

htc/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name htc/batch_normalization_2/beta

2htc/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_2/beta*
_output_shapes	
:*
dtype0

htc/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!htc/batch_normalization_2/gamma

3htc/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOphtc/batch_normalization_2/gamma*
_output_shapes	
:*
dtype0
{
htc/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namehtc/conv2d_2/bias
t
%htc/conv2d_2/bias/Read/ReadVariableOpReadVariableOphtc/conv2d_2/bias*
_output_shapes	
:*
dtype0

htc/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namehtc/conv2d_2/kernel

'htc/conv2d_2/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_2/kernel*'
_output_shapes
:@*
dtype0
ª
)htc/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)htc/batch_normalization_1/moving_variance
£
=htc/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp)htc/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
¢
%htc/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%htc/batch_normalization_1/moving_mean

9htc/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp%htc/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0

htc/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name htc/batch_normalization_1/beta

2htc/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization_1/beta*
_output_shapes
:@*
dtype0

htc/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!htc/batch_normalization_1/gamma

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

htc/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_namehtc/conv2d_1/kernel

'htc/conv2d_1/kernel/Read/ReadVariableOpReadVariableOphtc/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
¦
'htc/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'htc/batch_normalization/moving_variance

;htc/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp'htc/batch_normalization/moving_variance*
_output_shapes
: *
dtype0

#htc/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#htc/batch_normalization/moving_mean

7htc/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp#htc/batch_normalization/moving_mean*
_output_shapes
: *
dtype0

htc/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namehtc/batch_normalization/beta

0htc/batch_normalization/beta/Read/ReadVariableOpReadVariableOphtc/batch_normalization/beta*
_output_shapes
: *
dtype0

htc/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namehtc/batch_normalization/gamma

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

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

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿôô
Ï
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1htc/conv2d/kernelhtc/conv2d/biashtc/batch_normalization/gammahtc/batch_normalization/beta#htc/batch_normalization/moving_mean'htc/batch_normalization/moving_variancehtc/conv2d_1/kernelhtc/conv2d_1/biashtc/batch_normalization_1/gammahtc/batch_normalization_1/beta%htc/batch_normalization_1/moving_mean)htc/batch_normalization_1/moving_variancehtc/conv2d_2/kernelhtc/conv2d_2/biashtc/batch_normalization_2/gammahtc/batch_normalization_2/beta%htc/batch_normalization_2/moving_mean)htc/batch_normalization_2/moving_variancehtc/conv2d_3/kernelhtc/conv2d_3/biashtc/batch_normalization_3/gammahtc/batch_normalization_3/beta%htc/batch_normalization_3/moving_mean)htc/batch_normalization_3/moving_variancehtc/conv2d_4/kernelhtc/conv2d_4/biashtc/batch_normalization_4/gammahtc/batch_normalization_4/beta%htc/batch_normalization_4/moving_mean)htc/batch_normalization_4/moving_variancehtc/dense/kernelhtc/dense/bias%htc/batch_normalization_5/moving_mean)htc/batch_normalization_5/moving_variancehtc/batch_normalization_5/betahtc/batch_normalization_5/gamma*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_297449

NoOpNoOp
Í×
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*×
valueüÖBøÖ BðÖ
õ
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
Ú
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
º
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
°
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

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
È
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

1kernel
2bias
 y_jit_compiled_convolution_op*

z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	3gamma
4beta
5moving_mean
6moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

7kernel
8bias
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	 axis
	9gamma
:beta
;moving_mean
<moving_variance*

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses* 
Ï
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses

=kernel
>bias
!­_jit_compiled_convolution_op*

®	variables
¯trainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses* 
Ü
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
	ºaxis
	?gamma
@beta
Amoving_mean
Bmoving_variance*

»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¿__call__
+À&call_and_return_all_conditional_losses* 
Ï
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses

Ckernel
Dbias
!Ç_jit_compiled_convolution_op*

È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses* 
Ü
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses
	Ôaxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance*

Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses* 
Ï
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses

Ikernel
Jbias
!á_jit_compiled_convolution_op*

â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses* 
Ü
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
ì__call__
+í&call_and_return_all_conditional_losses
	îaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance*

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses* 
¬
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses
û_random_generator* 

ü	variables
ýtrainable_variables
þregularization_losses
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Okernel
Pbias*
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

trace_0* 

trace_0* 

serving_default* 
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

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
Â
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
²
c0
1
2
3
4
5
6
7
8
 9
¡10
¢11
£12
¤13
¥14
¦15
§16
¨17
©18
ª19
«20
¬21
­22
®23
¯24
°25
±26
²27
³28
´29
µ30
¶31
·32
¸33
¹34
º35
»36
¼37
½38
¾39
¿40
À41
Á42
Â43
Ã44
Ä45
Å46
Æ47
Ç48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ò
0
1
2
3
 4
¢5
¤6
¦7
¨8
ª9
¬10
®11
°12
²13
´14
¶15
¸16
º17
¼18
¾19
À20
Â21
Ä22
Æ23*
Ò
0
1
2
3
¡4
£5
¥6
§7
©8
«9
­10
¯11
±12
³13
µ14
·15
¹16
»17
½18
¿19
Á20
Ã21
Å22
Ç23*
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

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

Ítrace_0* 

Îtrace_0* 
* 
* 
* 
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ôtrace_0* 

Õtrace_0* 
 
30
41
52
63*

30
41*
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ûtrace_0
Ütrace_1* 

Ýtrace_0
Þtrace_1* 
* 
* 
* 
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ätrace_0* 

åtrace_0* 

70
81*

70
81*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ëtrace_0* 

ìtrace_0* 
* 
* 
* 
* 

ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

òtrace_0* 

ótrace_0* 
 
90
:1
;2
<3*

90
:1*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ùtrace_0
útrace_1* 

ûtrace_0
ütrace_1* 
* 
* 
* 
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

=0
>1*

=0
>1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
®	variables
¯trainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
 
?0
@1
A2
B3*

?0
@1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
»	variables
¼trainable_variables
½regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses* 

 trace_0* 

¡trace_0* 

C0
D1*

C0
D1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses*

§trace_0* 

¨trace_0* 
* 
* 
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses* 

®trace_0* 

¯trace_0* 
 
E0
F1
G2
H3*

E0
F1*
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses*

µtrace_0
¶trace_1* 

·trace_0
¸trace_1* 
* 
* 
* 
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses* 

¾trace_0* 

¿trace_0* 

I0
J1*

I0
J1*
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses*

Åtrace_0* 

Ætrace_0* 
* 
* 
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
â	variables
ãtrainable_variables
äregularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses* 

Ìtrace_0* 

Ítrace_0* 
 
K0
L1
M2
N3*

K0
L1*
* 

Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
è	variables
étrainable_variables
êregularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses*

Ótrace_0
Ôtrace_1* 

Õtrace_0
Ötrace_1* 
* 
* 
* 
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses* 

Ütrace_0* 

Ýtrace_0* 
* 
* 
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses* 

ãtrace_0
ätrace_1* 

åtrace_0
ætrace_1* 
* 
* 
* 
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
ü	variables
ýtrainable_variables
þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ìtrace_0* 

ítrace_0* 

O0
P1*

O0
P1*
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ótrace_0* 

ôtrace_0* 
 
Q0
R1
S2
T3*

Q0
R1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

útrace_0
ûtrace_1* 

ütrace_0
ýtrace_1* 
* 
* 
* 
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
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
¤'
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
GPU2 *0J 8 *(
f#R!
__inference__traced_save_299020
Ç
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
GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_299312»Ð#
 "
Õ
#__inference__update_step_xla_296248
gradient"
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: 7
sub_2_readvariableop_resource: 7
sub_3_readvariableop_resource: ¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=Z
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*&
_output_shapes
: 
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
 *o:Z
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*&
_output_shapes
: 
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*&
_output_shapes
: *
dtype0d
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*&
_output_shapes
: 
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
 *¿Ö3]
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
Ú 
´
#__inference__update_step_xla_297049
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ú 
´
#__inference__update_step_xla_296908
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

Ä
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298409

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
L
0__inference_max_pooling2d_4_layer_call_fn_298342

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_294447
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_298347

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
_
C__inference_flatten_layer_call_and_return_conditional_losses_294785

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ï
4__inference_batch_normalization_layer_call_fn_297969

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294199
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_298088

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_294143

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ±
Æ
?__inference_htc_layer_call_and_return_conditional_losses_297748
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
'conv2d_2_conv2d_readvariableop_resource:@7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_4_conv2d_readvariableop_resource:7
(conv2d_4_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	7
$dense_matmul_readvariableop_resource:	$3
%dense_biasadd_readvariableop_resource:@
2batch_normalization_5_cast_readvariableop_resource:B
4batch_normalization_5_cast_1_readvariableop_resource:B
4batch_normalization_5_cast_2_readvariableop_resource:B
4batch_normalization_5_cast_3_readvariableop_resource:
identity¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢5batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_4/ReadVariableOp¢&batch_normalization_4/ReadVariableOp_1¢)batch_normalization_5/Cast/ReadVariableOp¢+batch_normalization_5/Cast_1/ReadVariableOp¢+batch_normalization_5/Cast_2/ReadVariableOp¢+batch_normalization_5/Cast_3/ReadVariableOp¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¥
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí ¦
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
ksize
*
paddingVALID*
strides

"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0²
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿbb : : : : :*
epsilon%o:*
is_training( v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¾
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@ª
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¾
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Á
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( {
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Á
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( {
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Á
conv2d_4/Conv2DConv2Dre_lu_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ã
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( {
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transposere_lu_4/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dropout/IdentityIdentitytranspose:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
transpose_1	Transposedropout/Identity:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   v
flatten/ReshapeReshapetranspose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	$*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0
+batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0
+batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
#batch_normalization_5/batchnorm/addAddV23batch_normalization_5/Cast_1/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:¯
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 
%batch_normalization_5/batchnorm/mul_1Muldense/Softmax:softmax:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
%batch_normalization_5/batchnorm/mul_2Mul1batch_normalization_5/Cast/ReadVariableOp:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:¯
#batch_normalization_5/batchnorm/subSub3batch_normalization_5/Cast_2/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:´
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp,^batch_normalization_5/Cast_2/ReadVariableOp,^batch_normalization_5/Cast_3/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:ÿÿÿÿÿÿÿÿÿôô

_user_specified_namex
^
£
#__inference_internal_grad_fn_298906
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
:@L

Identity_9Identityresult_grads_9*
T0*
_output_shapes	
:N
Identity_10Identityresult_grads_10*
T0*
_output_shapes	
:N
Identity_11Identityresult_grads_11*
T0*
_output_shapes	
:[
Identity_12Identityresult_grads_12*
T0*(
_output_shapes
:N
Identity_13Identityresult_grads_13*
T0*
_output_shapes	
:N
Identity_14Identityresult_grads_14*
T0*
_output_shapes	
:N
Identity_15Identityresult_grads_15*
T0*
_output_shapes	
:[
Identity_16Identityresult_grads_16*
T0*(
_output_shapes
:N
Identity_17Identityresult_grads_17*
T0*
_output_shapes	
:N
Identity_18Identityresult_grads_18*
T0*
_output_shapes	
:N
Identity_19Identityresult_grads_19*
T0*
_output_shapes	
:R
Identity_20Identityresult_grads_20*
T0*
_output_shapes
:	$M
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
:ò

	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_10result_grads_11result_grads_12result_grads_13result_grads_14result_grads_15result_grads_16result_grads_17result_grads_18result_grads_19result_grads_20result_grads_21result_grads_22result_grads_23result_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_10result_grads_11result_grads_12result_grads_13result_grads_14result_grads_15result_grads_16result_grads_17result_grads_18result_grads_19result_grads_20result_grads_21result_grads_22result_grads_23*9
T4
220*,
_gradient_op_typeCustomGradient-298809*Ô
_output_shapesÁ
¾: : : : : @:@:@:@:@::::::::::::	$:::: : : : : @:@:@:@:@::::::::::::	$:::\
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
:@Q
Identity_33IdentityIdentityN:output:9*
T0*
_output_shapes	
:R
Identity_34IdentityIdentityN:output:10*
T0*
_output_shapes	
:R
Identity_35IdentityIdentityN:output:11*
T0*
_output_shapes	
:_
Identity_36IdentityIdentityN:output:12*
T0*(
_output_shapes
:R
Identity_37IdentityIdentityN:output:13*
T0*
_output_shapes	
:R
Identity_38IdentityIdentityN:output:14*
T0*
_output_shapes	
:R
Identity_39IdentityIdentityN:output:15*
T0*
_output_shapes	
:_
Identity_40IdentityIdentityN:output:16*
T0*(
_output_shapes
:R
Identity_41IdentityIdentityN:output:17*
T0*
_output_shapes	
:R
Identity_42IdentityIdentityN:output:18*
T0*
_output_shapes	
:R
Identity_43IdentityIdentityN:output:19*
T0*
_output_shapes	
:V
Identity_44IdentityIdentityN:output:20*
T0*
_output_shapes
:	$Q
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
identity_47Identity_47:output:0*Ó
_input_shapesÁ
¾: : : : : @:@:@:@:@::::::::::::	$:::: : : : : @:@:@:@:@::::::::::::	$::::V R
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
:@
(
_user_specified_nameresult_grads_8:K	G

_output_shapes	
:
(
_user_specified_nameresult_grads_9:L
H

_output_shapes	
:
)
_user_specified_nameresult_grads_10:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_11:YU
(
_output_shapes
:
)
_user_specified_nameresult_grads_12:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_13:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_14:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_15:YU
(
_output_shapes
:
)
_user_specified_nameresult_grads_16:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_17:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_18:LH

_output_shapes	
:
)
_user_specified_nameresult_grads_19:PL

_output_shapes
:	$
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
:@
)
_user_specified_nameresult_grads_32:L!H

_output_shapes	
:
)
_user_specified_nameresult_grads_33:L"H

_output_shapes	
:
)
_user_specified_nameresult_grads_34:L#H

_output_shapes	
:
)
_user_specified_nameresult_grads_35:Y$U
(
_output_shapes
:
)
_user_specified_nameresult_grads_36:L%H

_output_shapes	
:
)
_user_specified_nameresult_grads_37:L&H

_output_shapes	
:
)
_user_specified_nameresult_grads_38:L'H

_output_shapes	
:
)
_user_specified_nameresult_grads_39:Y(U
(
_output_shapes
:
)
_user_specified_nameresult_grads_40:L)H

_output_shapes	
:
)
_user_specified_nameresult_grads_41:L*H

_output_shapes	
:
)
_user_specified_nameresult_grads_42:L+H

_output_shapes	
:
)
_user_specified_nameresult_grads_43:P,L

_output_shapes
:	$
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
ë
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_298419

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_297943

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿ã
Â$
__inference_test_step_295760

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
+htc_conv2d_2_conv2d_readvariableop_resource:@;
,htc_conv2d_2_biasadd_readvariableop_resource:	@
1htc_batch_normalization_2_readvariableop_resource:	B
3htc_batch_normalization_2_readvariableop_1_resource:	Q
Bhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	G
+htc_conv2d_3_conv2d_readvariableop_resource:;
,htc_conv2d_3_biasadd_readvariableop_resource:	@
1htc_batch_normalization_3_readvariableop_resource:	B
3htc_batch_normalization_3_readvariableop_1_resource:	Q
Bhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	G
+htc_conv2d_4_conv2d_readvariableop_resource:;
,htc_conv2d_4_biasadd_readvariableop_resource:	@
1htc_batch_normalization_4_readvariableop_resource:	B
3htc_batch_normalization_4_readvariableop_1_resource:	Q
Bhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	;
(htc_dense_matmul_readvariableop_resource:	$7
)htc_dense_biasadd_readvariableop_resource:D
6htc_batch_normalization_5_cast_readvariableop_resource:F
8htc_batch_normalization_5_cast_1_readvariableop_resource:F
8htc_batch_normalization_5_cast_2_readvariableop_resource:F
8htc_batch_normalization_5_cast_3_readvariableop_resource:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: ¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignAddVariableOp_2¢AssignAddVariableOp_3¢div_no_nan/ReadVariableOp¢div_no_nan/ReadVariableOp_1¢div_no_nan_1/ReadVariableOp¢div_no_nan_1/ReadVariableOp_1¢7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp¢9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢&htc/batch_normalization/ReadVariableOp¢(htc/batch_normalization/ReadVariableOp_1¢9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_1/ReadVariableOp¢*htc/batch_normalization_1/ReadVariableOp_1¢9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_2/ReadVariableOp¢*htc/batch_normalization_2/ReadVariableOp_1¢9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_3/ReadVariableOp¢*htc/batch_normalization_3/ReadVariableOp_1¢9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_4/ReadVariableOp¢*htc/batch_normalization_4/ReadVariableOp_1¢-htc/batch_normalization_5/Cast/ReadVariableOp¢/htc/batch_normalization_5/Cast_1/ReadVariableOp¢/htc/batch_normalization_5/Cast_2/ReadVariableOp¢/htc/batch_normalization_5/Cast_3/ReadVariableOp¢!htc/conv2d/BiasAdd/ReadVariableOp¢ htc/conv2d/Conv2D/ReadVariableOp¢#htc/conv2d_1/BiasAdd/ReadVariableOp¢"htc/conv2d_1/Conv2D/ReadVariableOp¢#htc/conv2d_2/BiasAdd/ReadVariableOp¢"htc/conv2d_2/Conv2D/ReadVariableOp¢#htc/conv2d_3/BiasAdd/ReadVariableOp¢"htc/conv2d_3/Conv2D/ReadVariableOp¢#htc/conv2d_4/BiasAdd/ReadVariableOp¢"htc/conv2d_4/Conv2D/ReadVariableOp¢ htc/dense/BiasAdd/ReadVariableOp¢htc/dense/MatMul/ReadVariableOpZ
htc/CastCastimages*

DstT0*

SrcT0*(
_output_shapes
: ôô
 htc/conv2d/Conv2D/ReadVariableOpReadVariableOp)htc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¯
htc/conv2d/Conv2DConv2Dhtc/Cast:y:0(htc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
: íí *
paddingVALID*
strides

!htc/conv2d/BiasAdd/ReadVariableOpReadVariableOp*htc_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
htc/conv2d/BiasAddBiasAddhtc/conv2d/Conv2D:output:0)htc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
: íí ¥
htc/max_pooling2d/MaxPoolMaxPoolhtc/conv2d/BiasAdd:output:0*&
_output_shapes
: bb *
ksize
*
paddingVALID*
strides

&htc/batch_normalization/ReadVariableOpReadVariableOp/htc_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
(htc/batch_normalization/ReadVariableOp_1ReadVariableOp1htc_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Á
(htc/batch_normalization/FusedBatchNormV3FusedBatchNormV3"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:00htc/batch_normalization/ReadVariableOp_1:value:0?htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ahtc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: bb : : : : :*
epsilon%o:*
is_training( u
htc/re_lu/ReluRelu,htc/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: bb 
"htc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Á
htc/conv2d_1/Conv2DConv2Dhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@*
paddingVALID*
strides

#htc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
htc/conv2d_1/BiasAddBiasAddhtc/conv2d_1/Conv2D:output:0+htc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@©
htc/max_pooling2d_1/MaxPoolMaxPoolhtc/conv2d_1/BiasAdd:output:0*&
_output_shapes
: @*
ksize
*
paddingVALID*
strides

(htc/batch_normalization_1/ReadVariableOpReadVariableOp1htc_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
*htc/batch_normalization_1/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0¸
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¼
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Í
*htc/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:02htc/batch_normalization_1/ReadVariableOp_1:value:0Ahtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: @:@:@:@:@:*
epsilon%o:*
is_training( y
htc/re_lu_1/ReluRelu.htc/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: @
"htc/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ä
htc/conv2d_2/Conv2DConv2Dhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: *
paddingVALID*
strides

#htc/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
htc/conv2d_2/BiasAddBiasAddhtc/conv2d_2/Conv2D:output:0+htc/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ª
htc/max_pooling2d_2/MaxPoolMaxPoolhtc/conv2d_2/BiasAdd:output:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides

(htc/batch_normalization_2/ReadVariableOpReadVariableOp1htc_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_2/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ò
*htc/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:02htc/batch_normalization_2/ReadVariableOp_1:value:0Ahtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: :::::*
epsilon%o:*
is_training( z
htc/re_lu_2/ReluRelu.htc/batch_normalization_2/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: 
"htc/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
htc/conv2d_3/Conv2DConv2Dhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: *
paddingVALID*
strides

#htc/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
htc/conv2d_3/BiasAddBiasAddhtc/conv2d_3/Conv2D:output:0+htc/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ª
htc/max_pooling2d_3/MaxPoolMaxPoolhtc/conv2d_3/BiasAdd:output:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides

(htc/batch_normalization_3/ReadVariableOpReadVariableOp1htc_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_3/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ò
*htc/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:02htc/batch_normalization_3/ReadVariableOp_1:value:0Ahtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: :::::*
epsilon%o:*
is_training( z
htc/re_lu_3/ReluRelu.htc/batch_normalization_3/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: 
"htc/conv2d_4/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
htc/conv2d_4/Conv2DConv2Dhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: *
paddingVALID*
strides

#htc/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
htc/conv2d_4/BiasAddBiasAddhtc/conv2d_4/Conv2D:output:0+htc/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ª
htc/max_pooling2d_4/MaxPoolMaxPoolhtc/conv2d_4/BiasAdd:output:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides

(htc/batch_normalization_4/ReadVariableOpReadVariableOp1htc_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_4/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ò
*htc/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:02htc/batch_normalization_4/ReadVariableOp_1:value:0Ahtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: :::::*
epsilon%o:*
is_training( z
htc/re_lu_4/ReluRelu.htc/batch_normalization_4/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: k
htc/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
htc/transpose	Transposehtc/re_lu_4/Relu:activations:0htc/transpose/perm:output:0*
T0*'
_output_shapes
: e
htc/dropout/IdentityIdentityhtc/transpose:y:0*
T0*'
_output_shapes
: m
htc/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
htc/transpose_1	Transposehtc/dropout/Identity:output:0htc/transpose_1/perm:output:0*
T0*'
_output_shapes
: b
htc/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   y
htc/flatten/ReshapeReshapehtc/transpose_1:y:0htc/flatten/Const:output:0*
T0*
_output_shapes
:	 $
htc/dense/MatMul/ReadVariableOpReadVariableOp(htc_dense_matmul_readvariableop_resource*
_output_shapes
:	$*
dtype0
htc/dense/MatMulMatMulhtc/flatten/Reshape:output:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 htc/dense/BiasAdd/ReadVariableOpReadVariableOp)htc_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
htc/dense/BiasAddBiasAddhtc/dense/MatMul:product:0(htc/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: a
htc/dense/SoftmaxSoftmaxhtc/dense/BiasAdd:output:0*
T0*
_output_shapes

:  
-htc/batch_normalization_5/Cast/ReadVariableOpReadVariableOp6htc_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0n
)htc/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Â
'htc/batch_normalization_5/batchnorm/addAddV27htc/batch_normalization_5/Cast_1/ReadVariableOp:value:02htc/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
)htc/batch_normalization_5/batchnorm/RsqrtRsqrt+htc/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:»
'htc/batch_normalization_5/batchnorm/mulMul-htc/batch_normalization_5/batchnorm/Rsqrt:y:07htc/batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:£
)htc/batch_normalization_5/batchnorm/mul_1Mulhtc/dense/Softmax:softmax:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes

: ¹
)htc/batch_normalization_5/batchnorm/mul_2Mul5htc/batch_normalization_5/Cast/ReadVariableOp:value:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:»
'htc/batch_normalization_5/batchnorm/subSub7htc/batch_normalization_5/Cast_2/ReadVariableOp:value:0-htc/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
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
valueB"       
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits-htc/batch_normalization_5/batchnorm/add_1:z:0(sparse_categorical_crossentropy/Cast:y:0*
T0*$
_output_shapes
: : x
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ð
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: |
:sparse_categorical_crossentropy/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B : ¼
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
value	B :
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: Õ
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: ã
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
: 
SumSum7sparse_categorical_crossentropy/weighted_loss/value:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: ¡
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
: ·
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0
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
ÿÿÿÿÿÿÿÿÿr
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
: §
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
: »
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_4:y:0^AssignAddVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0¢
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2^AssignAddVariableOp_3*
_output_shapes
: *
dtype0
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0
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
j: ôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
: ôô
 
_user_specified_nameimages:B>

_output_shapes
: 
 
_user_specified_namelabels
ó

'__inference_conv2d_layer_call_fn_297923

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_294613y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôô: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
¡

ó
A__inference_dense_layer_call_and_return_conditional_losses_298477

inputs1
matmul_readvariableop_resource:	$-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_298246

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ½
Î!
!__inference__wrapped_model_294134
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
+htc_conv2d_2_conv2d_readvariableop_resource:@;
,htc_conv2d_2_biasadd_readvariableop_resource:	@
1htc_batch_normalization_2_readvariableop_resource:	B
3htc_batch_normalization_2_readvariableop_1_resource:	Q
Bhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	G
+htc_conv2d_3_conv2d_readvariableop_resource:;
,htc_conv2d_3_biasadd_readvariableop_resource:	@
1htc_batch_normalization_3_readvariableop_resource:	B
3htc_batch_normalization_3_readvariableop_1_resource:	Q
Bhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	G
+htc_conv2d_4_conv2d_readvariableop_resource:;
,htc_conv2d_4_biasadd_readvariableop_resource:	@
1htc_batch_normalization_4_readvariableop_resource:	B
3htc_batch_normalization_4_readvariableop_1_resource:	Q
Bhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	;
(htc_dense_matmul_readvariableop_resource:	$7
)htc_dense_biasadd_readvariableop_resource:D
6htc_batch_normalization_5_cast_readvariableop_resource:F
8htc_batch_normalization_5_cast_1_readvariableop_resource:F
8htc_batch_normalization_5_cast_2_readvariableop_resource:F
8htc_batch_normalization_5_cast_3_readvariableop_resource:
identity¢7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp¢9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢&htc/batch_normalization/ReadVariableOp¢(htc/batch_normalization/ReadVariableOp_1¢9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_1/ReadVariableOp¢*htc/batch_normalization_1/ReadVariableOp_1¢9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_2/ReadVariableOp¢*htc/batch_normalization_2/ReadVariableOp_1¢9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_3/ReadVariableOp¢*htc/batch_normalization_3/ReadVariableOp_1¢9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_4/ReadVariableOp¢*htc/batch_normalization_4/ReadVariableOp_1¢-htc/batch_normalization_5/Cast/ReadVariableOp¢/htc/batch_normalization_5/Cast_1/ReadVariableOp¢/htc/batch_normalization_5/Cast_2/ReadVariableOp¢/htc/batch_normalization_5/Cast_3/ReadVariableOp¢!htc/conv2d/BiasAdd/ReadVariableOp¢ htc/conv2d/Conv2D/ReadVariableOp¢#htc/conv2d_1/BiasAdd/ReadVariableOp¢"htc/conv2d_1/Conv2D/ReadVariableOp¢#htc/conv2d_2/BiasAdd/ReadVariableOp¢"htc/conv2d_2/Conv2D/ReadVariableOp¢#htc/conv2d_3/BiasAdd/ReadVariableOp¢"htc/conv2d_3/Conv2D/ReadVariableOp¢#htc/conv2d_4/BiasAdd/ReadVariableOp¢"htc/conv2d_4/Conv2D/ReadVariableOp¢ htc/dense/BiasAdd/ReadVariableOp¢htc/dense/MatMul/ReadVariableOp
 htc/conv2d/Conv2D/ReadVariableOpReadVariableOp)htc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0³
htc/conv2d/Conv2DConv2Dinput_1(htc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *
paddingVALID*
strides

!htc/conv2d/BiasAdd/ReadVariableOpReadVariableOp*htc_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
htc/conv2d/BiasAddBiasAddhtc/conv2d/Conv2D:output:0)htc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí ®
htc/max_pooling2d/MaxPoolMaxPoolhtc/conv2d/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
ksize
*
paddingVALID*
strides

&htc/batch_normalization/ReadVariableOpReadVariableOp/htc_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
(htc/batch_normalization/ReadVariableOp_1ReadVariableOp1htc_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ê
(htc/batch_normalization/FusedBatchNormV3FusedBatchNormV3"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:00htc/batch_normalization/ReadVariableOp_1:value:0?htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ahtc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿbb : : : : :*
epsilon%o:*
is_training( ~
htc/re_lu/ReluRelu,htc/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
"htc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ê
htc/conv2d_1/Conv2DConv2Dhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*
paddingVALID*
strides

#htc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
htc/conv2d_1/BiasAddBiasAddhtc/conv2d_1/Conv2D:output:0+htc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@²
htc/max_pooling2d_1/MaxPoolMaxPoolhtc/conv2d_1/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

(htc/batch_normalization_1/ReadVariableOpReadVariableOp1htc_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
*htc/batch_normalization_1/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0¸
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¼
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
*htc/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:02htc/batch_normalization_1/ReadVariableOp_1:value:0Ahtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
htc/re_lu_1/ReluRelu.htc/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"htc/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Í
htc/conv2d_2/Conv2DConv2Dhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

#htc/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
htc/conv2d_2/BiasAddBiasAddhtc/conv2d_2/Conv2D:output:0+htc/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
htc/max_pooling2d_2/MaxPoolMaxPoolhtc/conv2d_2/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

(htc/batch_normalization_2/ReadVariableOpReadVariableOp1htc_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_2/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
*htc/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:02htc/batch_normalization_2/ReadVariableOp_1:value:0Ahtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
htc/re_lu_2/ReluRelu.htc/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"htc/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
htc/conv2d_3/Conv2DConv2Dhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

#htc/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
htc/conv2d_3/BiasAddBiasAddhtc/conv2d_3/Conv2D:output:0+htc/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
htc/max_pooling2d_3/MaxPoolMaxPoolhtc/conv2d_3/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

(htc/batch_normalization_3/ReadVariableOpReadVariableOp1htc_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_3/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
*htc/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:02htc/batch_normalization_3/ReadVariableOp_1:value:0Ahtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
htc/re_lu_3/ReluRelu.htc/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"htc/conv2d_4/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
htc/conv2d_4/Conv2DConv2Dhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

#htc/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
htc/conv2d_4/BiasAddBiasAddhtc/conv2d_4/Conv2D:output:0+htc/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
htc/max_pooling2d_4/MaxPoolMaxPoolhtc/conv2d_4/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

(htc/batch_normalization_4/ReadVariableOpReadVariableOp1htc_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_4/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
*htc/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:02htc/batch_normalization_4/ReadVariableOp_1:value:0Ahtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
htc/re_lu_4/ReluRelu.htc/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
htc/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
htc/transpose	Transposehtc/re_lu_4/Relu:activations:0htc/transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
htc/dropout/IdentityIdentityhtc/transpose:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
htc/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
htc/transpose_1	Transposehtc/dropout/Identity:output:0htc/transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
htc/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
htc/flatten/ReshapeReshapehtc/transpose_1:y:0htc/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
htc/dense/MatMul/ReadVariableOpReadVariableOp(htc_dense_matmul_readvariableop_resource*
_output_shapes
:	$*
dtype0
htc/dense/MatMulMatMulhtc/flatten/Reshape:output:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 htc/dense/BiasAdd/ReadVariableOpReadVariableOp)htc_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
htc/dense/BiasAddBiasAddhtc/dense/MatMul:product:0(htc/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
htc/dense/SoftmaxSoftmaxhtc/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-htc/batch_normalization_5/Cast/ReadVariableOpReadVariableOp6htc_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0n
)htc/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Â
'htc/batch_normalization_5/batchnorm/addAddV27htc/batch_normalization_5/Cast_1/ReadVariableOp:value:02htc/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
)htc/batch_normalization_5/batchnorm/RsqrtRsqrt+htc/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:»
'htc/batch_normalization_5/batchnorm/mulMul-htc/batch_normalization_5/batchnorm/Rsqrt:y:07htc/batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:¬
)htc/batch_normalization_5/batchnorm/mul_1Mulhtc/dense/Softmax:softmax:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
)htc/batch_normalization_5/batchnorm/mul_2Mul5htc/batch_normalization_5/Cast/ReadVariableOp:value:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:»
'htc/batch_normalization_5/batchnorm/subSub7htc/batch_normalization_5/Cast_2/ReadVariableOp:value:0-htc/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:À
)htc/batch_normalization_5/batchnorm/add_1AddV2-htc/batch_normalization_5/batchnorm/mul_1:z:0+htc/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
htc/activation/SoftmaxSoftmax-htc/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
IdentityIdentity htc/activation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp8^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1'^htc/batch_normalization/ReadVariableOp)^htc/batch_normalization/ReadVariableOp_1:^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_1/ReadVariableOp+^htc/batch_normalization_1/ReadVariableOp_1:^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_2/ReadVariableOp+^htc/batch_normalization_2/ReadVariableOp_1:^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_3/ReadVariableOp+^htc/batch_normalization_3/ReadVariableOp_1:^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp<^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1)^htc/batch_normalization_4/ReadVariableOp+^htc/batch_normalization_4/ReadVariableOp_1.^htc/batch_normalization_5/Cast/ReadVariableOp0^htc/batch_normalization_5/Cast_1/ReadVariableOp0^htc/batch_normalization_5/Cast_2/ReadVariableOp0^htc/batch_normalization_5/Cast_3/ReadVariableOp"^htc/conv2d/BiasAdd/ReadVariableOp!^htc/conv2d/Conv2D/ReadVariableOp$^htc/conv2d_1/BiasAdd/ReadVariableOp#^htc/conv2d_1/Conv2D/ReadVariableOp$^htc/conv2d_2/BiasAdd/ReadVariableOp#^htc/conv2d_2/Conv2D/ReadVariableOp$^htc/conv2d_3/BiasAdd/ReadVariableOp#^htc/conv2d_3/Conv2D/ReadVariableOp$^htc/conv2d_4/BiasAdd/ReadVariableOp#^htc/conv2d_4/Conv2D/ReadVariableOp!^htc/dense/BiasAdd/ReadVariableOp ^htc/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
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
:ÿÿÿÿÿÿÿÿÿôô
!
_user_specified_name	input_1
Ü
 
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294472

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ü9
__inference_train_step_297370

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
+htc_conv2d_2_conv2d_readvariableop_resource:@;
,htc_conv2d_2_biasadd_readvariableop_resource:	@
1htc_batch_normalization_2_readvariableop_resource:	B
3htc_batch_normalization_2_readvariableop_1_resource:	Q
Bhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	G
+htc_conv2d_3_conv2d_readvariableop_resource:;
,htc_conv2d_3_biasadd_readvariableop_resource:	@
1htc_batch_normalization_3_readvariableop_resource:	B
3htc_batch_normalization_3_readvariableop_1_resource:	Q
Bhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	G
+htc_conv2d_4_conv2d_readvariableop_resource:;
,htc_conv2d_4_biasadd_readvariableop_resource:	@
1htc_batch_normalization_4_readvariableop_resource:	B
3htc_batch_normalization_4_readvariableop_1_resource:	Q
Bhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	S
Dhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	;
(htc_dense_matmul_readvariableop_resource:	$7
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

unknown_17:@%

unknown_18:@

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:&

unknown_26:

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	&

unknown_33:&

unknown_34:

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	

unknown_41:	$

unknown_42:	$

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:(
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: (
assignaddvariableop_4_resource: ¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignAddVariableOp_2¢AssignAddVariableOp_3¢AssignAddVariableOp_4¢StatefulPartitionedCall¢StatefulPartitionedCall_1¢StatefulPartitionedCall_10¢StatefulPartitionedCall_11¢StatefulPartitionedCall_12¢StatefulPartitionedCall_13¢StatefulPartitionedCall_14¢StatefulPartitionedCall_15¢StatefulPartitionedCall_16¢StatefulPartitionedCall_17¢StatefulPartitionedCall_18¢StatefulPartitionedCall_19¢StatefulPartitionedCall_2¢StatefulPartitionedCall_20¢StatefulPartitionedCall_21¢StatefulPartitionedCall_22¢StatefulPartitionedCall_23¢StatefulPartitionedCall_3¢StatefulPartitionedCall_4¢StatefulPartitionedCall_5¢StatefulPartitionedCall_6¢StatefulPartitionedCall_7¢StatefulPartitionedCall_8¢StatefulPartitionedCall_9¢div_no_nan/ReadVariableOp¢div_no_nan/ReadVariableOp_1¢div_no_nan_1/ReadVariableOp¢div_no_nan_1/ReadVariableOp_1¢&htc/batch_normalization/AssignNewValue¢(htc/batch_normalization/AssignNewValue_1¢7htc/batch_normalization/FusedBatchNormV3/ReadVariableOp¢9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢&htc/batch_normalization/ReadVariableOp¢(htc/batch_normalization/ReadVariableOp_1¢(htc/batch_normalization_1/AssignNewValue¢*htc/batch_normalization_1/AssignNewValue_1¢9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_1/ReadVariableOp¢*htc/batch_normalization_1/ReadVariableOp_1¢(htc/batch_normalization_2/AssignNewValue¢*htc/batch_normalization_2/AssignNewValue_1¢9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_2/ReadVariableOp¢*htc/batch_normalization_2/ReadVariableOp_1¢(htc/batch_normalization_3/AssignNewValue¢*htc/batch_normalization_3/AssignNewValue_1¢9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_3/ReadVariableOp¢*htc/batch_normalization_3/ReadVariableOp_1¢(htc/batch_normalization_4/AssignNewValue¢*htc/batch_normalization_4/AssignNewValue_1¢9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢(htc/batch_normalization_4/ReadVariableOp¢*htc/batch_normalization_4/ReadVariableOp_1¢)htc/batch_normalization_5/AssignMovingAvg¢8htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp¢+htc/batch_normalization_5/AssignMovingAvg_1¢:htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢-htc/batch_normalization_5/Cast/ReadVariableOp¢/htc/batch_normalization_5/Cast_1/ReadVariableOp¢!htc/conv2d/BiasAdd/ReadVariableOp¢ htc/conv2d/Conv2D/ReadVariableOp¢#htc/conv2d_1/BiasAdd/ReadVariableOp¢"htc/conv2d_1/Conv2D/ReadVariableOp¢#htc/conv2d_2/BiasAdd/ReadVariableOp¢"htc/conv2d_2/Conv2D/ReadVariableOp¢#htc/conv2d_3/BiasAdd/ReadVariableOp¢"htc/conv2d_3/Conv2D/ReadVariableOp¢#htc/conv2d_4/BiasAdd/ReadVariableOp¢"htc/conv2d_4/Conv2D/ReadVariableOp¢ htc/dense/BiasAdd/ReadVariableOp¢htc/dense/MatMul/ReadVariableOpZ
htc/CastCastimages*

DstT0*

SrcT0*(
_output_shapes
: ôô
 htc/conv2d/Conv2D/ReadVariableOpReadVariableOp)htc_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¯
htc/conv2d/Conv2DConv2Dhtc/Cast:y:0(htc/conv2d/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
: íí *
paddingVALID*
strides

!htc/conv2d/BiasAdd/ReadVariableOpReadVariableOp*htc_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
htc/conv2d/BiasAddBiasAddhtc/conv2d/Conv2D:output:0)htc/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
: íí ¥
htc/max_pooling2d/MaxPoolMaxPoolhtc/conv2d/BiasAdd:output:0*&
_output_shapes
: bb *
ksize
*
paddingVALID*
strides

&htc/batch_normalization/ReadVariableOpReadVariableOp/htc_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
(htc/batch_normalization/ReadVariableOp_1ReadVariableOp1htc_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0´
7htc/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¸
9htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ï
(htc/batch_normalization/FusedBatchNormV3FusedBatchNormV3"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:00htc/batch_normalization/ReadVariableOp_1:value:0?htc/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ahtc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: bb : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<¦
&htc/batch_normalization/AssignNewValueAssignVariableOp@htc_batch_normalization_fusedbatchnormv3_readvariableop_resource5htc/batch_normalization/FusedBatchNormV3:batch_mean:08^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
(htc/batch_normalization/AssignNewValue_1AssignVariableOpBhtc_batch_normalization_fusedbatchnormv3_readvariableop_1_resource9htc/batch_normalization/FusedBatchNormV3:batch_variance:0:^htc/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(u
htc/re_lu/ReluRelu,htc/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: bb 
"htc/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Á
htc/conv2d_1/Conv2DConv2Dhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@*
paddingVALID*
strides

#htc/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
htc/conv2d_1/BiasAddBiasAddhtc/conv2d_1/Conv2D:output:0+htc/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: ^^@©
htc/max_pooling2d_1/MaxPoolMaxPoolhtc/conv2d_1/BiasAdd:output:0*&
_output_shapes
: @*
ksize
*
paddingVALID*
strides

(htc/batch_normalization_1/ReadVariableOpReadVariableOp1htc_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
*htc/batch_normalization_1/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0¸
9htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¼
;htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Û
*htc/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:02htc/batch_normalization_1/ReadVariableOp_1:value:0Ahtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: @:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<®
(htc/batch_normalization_1/AssignNewValueAssignVariableOpBhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_1/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¸
*htc/batch_normalization_1/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_1/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(y
htc/re_lu_1/ReluRelu.htc/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: @
"htc/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ä
htc/conv2d_2/Conv2DConv2Dhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: *
paddingVALID*
strides

#htc/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
htc/conv2d_2/BiasAddBiasAddhtc/conv2d_2/Conv2D:output:0+htc/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ª
htc/max_pooling2d_2/MaxPoolMaxPoolhtc/conv2d_2/BiasAdd:output:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides

(htc/batch_normalization_2/ReadVariableOpReadVariableOp1htc_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_2/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0à
*htc/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:02htc/batch_normalization_2/ReadVariableOp_1:value:0Ahtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: :::::*
epsilon%o:*
exponential_avg_factor%
×#<®
(htc/batch_normalization_2/AssignNewValueAssignVariableOpBhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_2/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¸
*htc/batch_normalization_2/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_2/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
htc/re_lu_2/ReluRelu.htc/batch_normalization_2/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: 
"htc/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
htc/conv2d_3/Conv2DConv2Dhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: *
paddingVALID*
strides

#htc/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
htc/conv2d_3/BiasAddBiasAddhtc/conv2d_3/Conv2D:output:0+htc/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ª
htc/max_pooling2d_3/MaxPoolMaxPoolhtc/conv2d_3/BiasAdd:output:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides

(htc/batch_normalization_3/ReadVariableOpReadVariableOp1htc_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_3/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0à
*htc/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:02htc/batch_normalization_3/ReadVariableOp_1:value:0Ahtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: :::::*
epsilon%o:*
exponential_avg_factor%
×#<®
(htc/batch_normalization_3/AssignNewValueAssignVariableOpBhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_3/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¸
*htc/batch_normalization_3/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_3/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
htc/re_lu_3/ReluRelu.htc/batch_normalization_3/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: 
"htc/conv2d_4/Conv2D/ReadVariableOpReadVariableOp+htc_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
htc/conv2d_4/Conv2DConv2Dhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
: *
paddingVALID*
strides

#htc/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp,htc_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
htc/conv2d_4/BiasAddBiasAddhtc/conv2d_4/Conv2D:output:0+htc/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
: ª
htc/max_pooling2d_4/MaxPoolMaxPoolhtc/conv2d_4/BiasAdd:output:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides

(htc/batch_normalization_4/ReadVariableOpReadVariableOp1htc_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
*htc/batch_normalization_4/ReadVariableOp_1ReadVariableOp3htc_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0¹
9htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0½
;htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0à
*htc/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:02htc/batch_normalization_4/ReadVariableOp_1:value:0Ahtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Chtc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3: :::::*
epsilon%o:*
exponential_avg_factor%
×#<®
(htc/batch_normalization_4/AssignNewValueAssignVariableOpBhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_resource7htc/batch_normalization_4/FusedBatchNormV3:batch_mean:0:^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¸
*htc/batch_normalization_4/AssignNewValue_1AssignVariableOpDhtc_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource;htc/batch_normalization_4/FusedBatchNormV3:batch_variance:0<^htc/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
htc/re_lu_4/ReluRelu.htc/batch_normalization_4/FusedBatchNormV3:y:0*
T0*'
_output_shapes
: k
htc/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
htc/transpose	Transposehtc/re_lu_4/Relu:activations:0htc/transpose/perm:output:0*
T0*'
_output_shapes
: ^
htc/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
htc/dropout/dropout/MulMulhtc/transpose:y:0"htc/dropout/dropout/Const:output:0*
T0*'
_output_shapes
: r
htc/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
0htc/dropout/dropout/random_uniform/RandomUniformRandomUniform"htc/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
: *
dtype0g
"htc/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ê
 htc/dropout/dropout/GreaterEqualGreaterEqual9htc/dropout/dropout/random_uniform/RandomUniform:output:0+htc/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
: `
htc/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ã
htc/dropout/dropout/SelectV2SelectV2$htc/dropout/dropout/GreaterEqual:z:0htc/dropout/dropout/Mul:z:0$htc/dropout/dropout/Const_1:output:0*
T0*'
_output_shapes
: m
htc/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
htc/transpose_1	Transpose%htc/dropout/dropout/SelectV2:output:0htc/transpose_1/perm:output:0*
T0*'
_output_shapes
: b
htc/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   y
htc/flatten/ReshapeReshapehtc/transpose_1:y:0htc/flatten/Const:output:0*
T0*
_output_shapes
:	 $
htc/dense/MatMul/ReadVariableOpReadVariableOp(htc_dense_matmul_readvariableop_resource*
_output_shapes
:	$*
dtype0
htc/dense/MatMulMatMulhtc/flatten/Reshape:output:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 htc/dense/BiasAdd/ReadVariableOpReadVariableOp)htc_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
htc/dense/BiasAddBiasAddhtc/dense/MatMul:product:0(htc/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: a
htc/dense/SoftmaxSoftmaxhtc/dense/BiasAdd:output:0*
T0*
_output_shapes

: 
8htc/batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: È
&htc/batch_normalization_5/moments/meanMeanhtc/dense/Softmax:softmax:0Ahtc/batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
.htc/batch_normalization_5/moments/StopGradientStopGradient/htc/batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:Ç
3htc/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencehtc/dense/Softmax:softmax:07htc/batch_normalization_5/moments/StopGradient:output:0*
T0*
_output_shapes

: 
<htc/batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ì
*htc/batch_normalization_5/moments/varianceMean7htc/batch_normalization_5/moments/SquaredDifference:z:0Ehtc/batch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(¡
)htc/batch_normalization_5/moments/SqueezeSqueeze/htc/batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 §
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
×#<¶
8htc/batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOpAhtc_batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-htc/batch_normalization_5/AssignMovingAvg/subSub@htc/batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:02htc/batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:Æ
-htc/batch_normalization_5/AssignMovingAvg/mulMul1htc/batch_normalization_5/AssignMovingAvg/sub:z:08htc/batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
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
×#<º
:htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOpChtc_batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Õ
/htc/batch_normalization_5/AssignMovingAvg_1/subSubBhtc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:04htc/batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ì
/htc/batch_normalization_5/AssignMovingAvg_1/mulMul3htc/batch_normalization_5/AssignMovingAvg_1/sub:z:0:htc/batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
+htc/batch_normalization_5/AssignMovingAvg_1AssignSubVariableOpChtc_batch_normalization_5_assignmovingavg_1_readvariableop_resource3htc/batch_normalization_5/AssignMovingAvg_1/mul:z:0;^htc/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0 
-htc/batch_normalization_5/Cast/ReadVariableOpReadVariableOp6htc_batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0¤
/htc/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp8htc_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0n
)htc/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
'htc/batch_normalization_5/batchnorm/addAddV24htc/batch_normalization_5/moments/Squeeze_1:output:02htc/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
)htc/batch_normalization_5/batchnorm/RsqrtRsqrt+htc/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:»
'htc/batch_normalization_5/batchnorm/mulMul-htc/batch_normalization_5/batchnorm/Rsqrt:y:07htc/batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:£
)htc/batch_normalization_5/batchnorm/mul_1Mulhtc/dense/Softmax:softmax:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes

: ¶
)htc/batch_normalization_5/batchnorm/mul_2Mul2htc/batch_normalization_5/moments/Squeeze:output:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:¹
'htc/batch_normalization_5/batchnorm/subSub5htc/batch_normalization_5/Cast/ReadVariableOp:value:0-htc/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
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
valueB"       
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits-htc/batch_normalization_5/batchnorm/add_1:z:0(sparse_categorical_crossentropy/Cast:y:0*
T0*$
_output_shapes
: : x
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ð
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: |
:sparse_categorical_crossentropy/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B : ¼
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
value	B :
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: Õ
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: ã
3sparse_categorical_crossentropy/weighted_loss/valueDivNoNan<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB Ê
Wgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsPgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÍ
Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanDivNoNanones:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/SumSumPgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan:z:0\gradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: 
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/ReshapeReshapeNgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum:output:0Pgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: «
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/NegNeg<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: 
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1DivNoNanIgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Neg:y:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2DivNoNanRgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1:z:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: Ð
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mulMulones:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: 
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1SumIgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mul:z:0\gradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: 
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape_1ReshapePgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: 
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/ReshapeReshapeRgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape:output:0Tgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: 
Agradient_tape/sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB 
@gradient_tape/sparse_categorical_crossentropy/weighted_loss/TileTileLgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape:output:0Jgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1ReshapeIgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile:output:0Tgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1TileNgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1:output:0Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: ö
?gradient_tape/sparse_categorical_crossentropy/weighted_loss/MulMulKgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1:output:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: «
`gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
\gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims
ExpandDimsCgradient_tape/sparse_categorical_crossentropy/weighted_loss/Mul:z:0igradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dim:output:0*
T0*
_output_shapes

: à
Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulMulegradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims:output:0rsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:backprop:0*
T0*
_output_shapes

: ¡
Pgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB"       
Pgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB:Ð
Mgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgsBroadcastGradientArgsYgradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s0:output:0Ygradient_tape/htc/batch_normalization_5/batchnorm/add_1/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
Mgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
;gradient_tape/htc/batch_normalization_5/batchnorm/add_1/SumSumYgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0Vgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:
Egradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?gradient_tape/htc/batch_normalization_5/batchnorm/add_1/ReshapeReshapeDgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Sum:output:0Ngradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape/shape:output:0*
T0*
_output_shapes
:ó
;gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/MulMulYgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes

: å
=gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Mul_1Mulhtc/dense/Softmax:softmax:0Ygradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0*
T0*
_output_shapes

: 
Mgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
;gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/SumSumAgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Mul_1:z:0Vgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Sum/reduction_indices:output:0*
T0*
_output_shapes
:
Egradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?gradient_tape/htc/batch_normalization_5/batchnorm/mul_1/ReshapeReshapeDgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Sum:output:0Ngradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Reshape/shape:output:0*
T0*
_output_shapes
:¯
9gradient_tape/htc/batch_normalization_5/batchnorm/sub/NegNegHgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0*
T0*
_output_shapes
:Ó
;gradient_tape/htc/batch_normalization_5/batchnorm/mul_2/MulMul=gradient_tape/htc/batch_normalization_5/batchnorm/sub/Neg:y:0+htc/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:Ü
=gradient_tape/htc/batch_normalization_5/batchnorm/mul_2/Mul_1Mul=gradient_tape/htc/batch_normalization_5/batchnorm/sub/Neg:y:02htc/batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:
5gradient_tape/htc/batch_normalization_5/moments/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ì
7gradient_tape/htc/batch_normalization_5/moments/ReshapeReshape?gradient_tape/htc/batch_normalization_5/batchnorm/mul_2/Mul:z:0>gradient_tape/htc/batch_normalization_5/moments/Shape:output:0*
T0*
_output_shapes

:Ç
AddNAddNHgradient_tape/htc/batch_normalization_5/batchnorm/mul_1/Reshape:output:0Agradient_tape/htc/batch_normalization_5/batchnorm/mul_2/Mul_1:z:0*
N*
T0*
_output_shapes
:ª
9gradient_tape/htc/batch_normalization_5/batchnorm/mul/MulMul
AddN:sum:07htc/batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:¢
;gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1Mul
AddN:sum:0-htc/batch_normalization_5/batchnorm/Rsqrt:y:0*
T0*
_output_shapes
:
9gradient_tape/htc/batch_normalization_5/moments/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"      {
9gradient_tape/htc/batch_normalization_5/moments/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :ï
7gradient_tape/htc/batch_normalization_5/moments/MaximumMaximumBgradient_tape/htc/batch_normalization_5/moments/Maximum/x:output:0Bgradient_tape/htc/batch_normalization_5/moments/Maximum/y:output:0*
T0*
_output_shapes
:
:gradient_tape/htc/batch_normalization_5/moments/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"       ë
8gradient_tape/htc/batch_normalization_5/moments/floordivFloorDivCgradient_tape/htc/batch_normalization_5/moments/floordiv/x:output:0;gradient_tape/htc/batch_normalization_5/moments/Maximum:z:0*
T0*
_output_shapes
:
?gradient_tape/htc/batch_normalization_5/moments/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ù
9gradient_tape/htc/batch_normalization_5/moments/Reshape_1Reshape@gradient_tape/htc/batch_normalization_5/moments/Reshape:output:0Hgradient_tape/htc/batch_normalization_5/moments/Reshape_1/shape:output:0*
T0*
_output_shapes

:
>gradient_tape/htc/batch_normalization_5/moments/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"       ò
4gradient_tape/htc/batch_normalization_5/moments/TileTileBgradient_tape/htc/batch_normalization_5/moments/Reshape_1:output:0Ggradient_tape/htc/batch_normalization_5/moments/Tile/multiples:output:0*
T0*
_output_shapes

: z
5gradient_tape/htc/batch_normalization_5/moments/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Bê
7gradient_tape/htc/batch_normalization_5/moments/truedivRealDiv=gradient_tape/htc/batch_normalization_5/moments/Tile:output:0>gradient_tape/htc/batch_normalization_5/moments/Const:output:0*
T0*
_output_shapes

: Û
;gradient_tape/htc/batch_normalization_5/batchnorm/RsqrtGrad	RsqrtGrad-htc/batch_normalization_5/batchnorm/Rsqrt:y:0=gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul:z:0*
T0*
_output_shapes
:
7gradient_tape/htc/batch_normalization_5/moments/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      ð
9gradient_tape/htc/batch_normalization_5/moments/Reshape_2Reshape?gradient_tape/htc/batch_normalization_5/batchnorm/RsqrtGrad:z:0@gradient_tape/htc/batch_normalization_5/moments/Shape_1:output:0*
T0*
_output_shapes

:
?gradient_tape/htc/batch_normalization_5/moments/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      û
9gradient_tape/htc/batch_normalization_5/moments/Reshape_3ReshapeBgradient_tape/htc/batch_normalization_5/moments/Reshape_2:output:0Hgradient_tape/htc/batch_normalization_5/moments/Reshape_3/shape:output:0*
T0*
_output_shapes

:
@gradient_tape/htc/batch_normalization_5/moments/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"       ö
6gradient_tape/htc/batch_normalization_5/moments/Tile_1TileBgradient_tape/htc/batch_normalization_5/moments/Reshape_3:output:0Igradient_tape/htc/batch_normalization_5/moments/Tile_1/multiples:output:0*
T0*
_output_shapes

: |
7gradient_tape/htc/batch_normalization_5/moments/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   B
9gradient_tape/htc/batch_normalization_5/moments/truediv_1RealDiv?gradient_tape/htc/batch_normalization_5/moments/Tile_1:output:0@gradient_tape/htc/batch_normalization_5/moments/Const_1:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes

: ·
6gradient_tape/htc/batch_normalization_5/moments/scalarConst:^gradient_tape/htc/batch_normalization_5/moments/truediv_1*
_output_shapes
: *
dtype0*
valueB
 *   @ã
3gradient_tape/htc/batch_normalization_5/moments/MulMul?gradient_tape/htc/batch_normalization_5/moments/scalar:output:0=gradient_tape/htc/batch_normalization_5/moments/truediv_1:z:0*
T0*
_output_shapes

: õ
3gradient_tape/htc/batch_normalization_5/moments/subSubhtc/dense/Softmax:softmax:07htc/batch_normalization_5/moments/StopGradient:output:0:^gradient_tape/htc/batch_normalization_5/moments/truediv_1*
T0*
_output_shapes

: ×
5gradient_tape/htc/batch_normalization_5/moments/mul_1Mul7gradient_tape/htc/batch_normalization_5/moments/Mul:z:07gradient_tape/htc/batch_normalization_5/moments/sub:z:0*
T0*
_output_shapes

: 
Hgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB"       
Hgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB"      ¸
Egradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgsBroadcastGradientArgsQgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s0:output:0Qgradient_tape/htc/batch_normalization_5/moments/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿù
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
ÿÿÿÿÿÿÿÿÿµ
gradient_tape/htc/dense/SumSumgradient_tape/htc/dense/mul:z:06gradient_tape/htc/dense/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(
gradient_tape/htc/dense/subSubAddN_1:sum:0$gradient_tape/htc/dense/Sum:output:0*
T0*
_output_shapes

: 
gradient_tape/htc/dense/mul_1Mulgradient_tape/htc/dense/sub:z:0htc/dense/Softmax:softmax:0*
T0*
_output_shapes

: 
+gradient_tape/htc/dense/BiasAdd/BiasAddGradBiasAddGrad!gradient_tape/htc/dense/mul_1:z:0*
T0*
_output_shapes
:¸
%gradient_tape/htc/dense/MatMul/MatMulMatMul!gradient_tape/htc/dense/mul_1:z:0'htc/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	 $*
transpose_b(¯
'gradient_tape/htc/dense/MatMul/MatMul_1MatMulhtc/flatten/Reshape:output:0!gradient_tape/htc/dense/mul_1:z:0*
T0*
_output_shapes
:	$*
transpose_a(x
gradient_tape/htc/flatten/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
!gradient_tape/htc/flatten/ReshapeReshape/gradient_tape/htc/dense/MatMul/MatMul:product:0(gradient_tape/htc/flatten/Shape:output:0*
T0*'
_output_shapes
: 
/gradient_tape/htc/transpose_1/InvertPermutationInvertPermutationhtc/transpose_1/perm:output:0*
_output_shapes
:Ç
'gradient_tape/htc/transpose_1/transpose	Transpose*gradient_tape/htc/flatten/Reshape:output:03gradient_tape/htc/transpose_1/InvertPermutation:y:0*
T0*'
_output_shapes
: l
'gradient_tape/htc/dropout/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    í
*gradient_tape/htc/dropout/dropout/SelectV2SelectV2$htc/dropout/dropout/GreaterEqual:z:0+gradient_tape/htc/transpose_1/transpose:y:00gradient_tape/htc/dropout/dropout/zeros:output:0*
T0*'
_output_shapes
: 
'gradient_tape/htc/dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             
)gradient_tape/htc/dropout/dropout/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"             ê
7gradient_tape/htc/dropout/dropout/BroadcastGradientArgsBroadcastGradientArgs0gradient_tape/htc/dropout/dropout/Shape:output:02gradient_tape/htc/dropout/dropout/Shape_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿâ
%gradient_tape/htc/dropout/dropout/SumSum3gradient_tape/htc/dropout/dropout/SelectV2:output:0<gradient_tape/htc/dropout/dropout/BroadcastGradientArgs:r0:0*
T0*'
_output_shapes
: *
	keep_dims(È
)gradient_tape/htc/dropout/dropout/ReshapeReshape.gradient_tape/htc/dropout/dropout/Sum:output:00gradient_tape/htc/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
: ï
,gradient_tape/htc/dropout/dropout/SelectV2_1SelectV2$htc/dropout/dropout/GreaterEqual:z:00gradient_tape/htc/dropout/dropout/zeros:output:0+gradient_tape/htc/transpose_1/transpose:y:0*
T0*'
_output_shapes
: l
)gradient_tape/htc/dropout/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB î
9gradient_tape/htc/dropout/dropout/BroadcastGradientArgs_1BroadcastGradientArgs2gradient_tape/htc/dropout/dropout/Shape_2:output:02gradient_tape/htc/dropout/dropout/Shape_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿç
'gradient_tape/htc/dropout/dropout/Sum_1Sum5gradient_tape/htc/dropout/dropout/SelectV2_1:output:0>gradient_tape/htc/dropout/dropout/BroadcastGradientArgs_1:r0:0*
T0*&
_output_shapes
:*
	keep_dims(½
+gradient_tape/htc/dropout/dropout/Reshape_1Reshape0gradient_tape/htc/dropout/dropout/Sum_1:output:02gradient_tape/htc/dropout/dropout/Shape_2:output:0*
T0*
_output_shapes
: ¶
%gradient_tape/htc/dropout/dropout/MulMul2gradient_tape/htc/dropout/dropout/Reshape:output:0"htc/dropout/dropout/Const:output:0*
T0*'
_output_shapes
: {
-gradient_tape/htc/transpose/InvertPermutationInvertPermutationhtc/transpose/perm:output:0*
_output_shapes
:Â
%gradient_tape/htc/transpose/transpose	Transpose)gradient_tape/htc/dropout/dropout/Mul:z:01gradient_tape/htc/transpose/InvertPermutation:y:0*
T0*'
_output_shapes
: «
"gradient_tape/htc/re_lu_4/ReluGradReluGrad)gradient_tape/htc/transpose/transpose:y:0htc/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
: T
zerosConst*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_1Const*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_2Const*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_3Const*
_output_shapes	
:*
dtype0*
valueB*    x

zeros_like	ZerosLike<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:ù
<gradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_4/ReluGrad:backprops:0$htc/max_pooling2d_4/MaxPool:output:00htc/batch_normalization_4/ReadVariableOp:value:0<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_4/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*=
_output_shapes+
): ::: : *
epsilon%o:¹
5gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_4/BiasAdd:output:0$htc/max_pooling2d_4/MaxPool:output:0Igradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:x_backprop:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides
£
.gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes	
:²
(gradient_tape/htc/conv2d_4/Conv2D/ShapeNShapeNhtc/re_lu_3/Relu:activations:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::Æ
5gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_4/Conv2D/ShapeN:output:0*htc/conv2d_4/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGrad:output:0*
T0*'
_output_shapes
: *
paddingVALID*
strides
½
6gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu_3/Relu:activations:01gradient_tape/htc/conv2d_4/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_4/MaxPool/MaxPoolGrad:output:0*
T0*(
_output_shapes
:*
paddingVALID*
strides
À
"gradient_tape/htc/re_lu_3/ReluGradReluGrad>gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropInput:output:0htc/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
: V
zeros_4Const*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_5Const*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_6Const*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_7Const*
_output_shapes	
:*
dtype0*
valueB*    z
zeros_like_1	ZerosLike<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:ù
<gradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_3/ReluGrad:backprops:0$htc/max_pooling2d_3/MaxPool:output:00htc/batch_normalization_3/ReadVariableOp:value:0<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_3/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*=
_output_shapes+
): ::: : *
epsilon%o:¹
5gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_3/BiasAdd:output:0$htc/max_pooling2d_3/MaxPool:output:0Igradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:x_backprop:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides
£
.gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes	
:²
(gradient_tape/htc/conv2d_3/Conv2D/ShapeNShapeNhtc/re_lu_2/Relu:activations:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::Æ
5gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_3/Conv2D/ShapeN:output:0*htc/conv2d_3/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGrad:output:0*
T0*'
_output_shapes
: *
paddingVALID*
strides
½
6gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu_2/Relu:activations:01gradient_tape/htc/conv2d_3/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_3/MaxPool/MaxPoolGrad:output:0*
T0*(
_output_shapes
:*
paddingVALID*
strides
À
"gradient_tape/htc/re_lu_2/ReluGradReluGrad>gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropInput:output:0htc/re_lu_2/Relu:activations:0*
T0*'
_output_shapes
: V
zeros_8Const*
_output_shapes	
:*
dtype0*
valueB*    V
zeros_9Const*
_output_shapes	
:*
dtype0*
valueB*    W
zeros_10Const*
_output_shapes	
:*
dtype0*
valueB*    W
zeros_11Const*
_output_shapes	
:*
dtype0*
valueB*    z
zeros_like_2	ZerosLike<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_3:0*
T0*
_output_shapes
:ù
<gradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_2/ReluGrad:backprops:0$htc/max_pooling2d_2/MaxPool:output:00htc/batch_normalization_2/ReadVariableOp:value:0<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_2/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*=
_output_shapes+
): ::: : *
epsilon%o:¹
5gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_2/BiasAdd:output:0$htc/max_pooling2d_2/MaxPool:output:0Igradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:x_backprop:0*'
_output_shapes
: *
ksize
*
paddingVALID*
strides
£
.gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes	
:²
(gradient_tape/htc/conv2d_2/Conv2D/ShapeNShapeNhtc/re_lu_1/Relu:activations:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::Å
5gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_2/Conv2D/ShapeN:output:0*htc/conv2d_2/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: @*
paddingVALID*
strides
¼
6gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu_1/Relu:activations:01gradient_tape/htc/conv2d_2/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_2/MaxPool/MaxPoolGrad:output:0*
T0*'
_output_shapes
:@*
paddingVALID*
strides
¿
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
:ö
<gradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3FusedBatchNormGradV3.gradient_tape/htc/re_lu_1/ReluGrad:backprops:0$htc/max_pooling2d_1/MaxPool:output:00htc/batch_normalization_1/ReadVariableOp:value:0<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_1:0<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_2:0<htc/batch_normalization_1/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*:
_output_shapes(
&: @:@:@: : *
epsilon%o:¸
5gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d_1/BiasAdd:output:0$htc/max_pooling2d_1/MaxPool:output:0Igradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:x_backprop:0*&
_output_shapes
: ^^@*
ksize
*
paddingVALID*
strides
¢
.gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes
:@°
(gradient_tape/htc/conv2d_1/Conv2D/ShapeNShapeNhtc/re_lu/Relu:activations:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::Å
5gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput1gradient_tape/htc/conv2d_1/Conv2D/ShapeN:output:0*htc/conv2d_1/Conv2D/ReadVariableOp:value:0>gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: bb *
paddingVALID*
strides
¹
6gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/re_lu/Relu:activations:01gradient_tape/htc/conv2d_1/Conv2D/ShapeN:output:1>gradient_tape/htc/max_pooling2d_1/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: @*
paddingVALID*
strides
»
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
:è
:gradient_tape/htc/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3,gradient_tape/htc/re_lu/ReluGrad:backprops:0"htc/max_pooling2d/MaxPool:output:0.htc/batch_normalization/ReadVariableOp:value:0:htc/batch_normalization/FusedBatchNormV3:reserve_space_1:0:htc/batch_normalization/FusedBatchNormV3:reserve_space_2:0:htc/batch_normalization/FusedBatchNormV3:reserve_space_3:0*
T0*
U0*:
_output_shapes(
&: bb : : : : *
epsilon%o:²
3gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGradhtc/conv2d/BiasAdd:output:0"htc/max_pooling2d/MaxPool:output:0Ggradient_tape/htc/batch_normalization/FusedBatchNormGradV3:x_backprop:0*(
_output_shapes
: íí *
ksize
*
paddingVALID*
strides

,gradient_tape/htc/conv2d/BiasAdd/BiasAddGradBiasAddGrad<gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGrad:output:0*
T0*
_output_shapes
: 
&gradient_tape/htc/conv2d/Conv2D/ShapeNShapeNhtc/Cast:y:0(htc/conv2d/Conv2D/ReadVariableOp:value:0*
N*
T0* 
_output_shapes
::¿
3gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropInputConv2DBackpropInput/gradient_tape/htc/conv2d/Conv2D/ShapeN:output:0(htc/conv2d/Conv2D/ReadVariableOp:value:0<gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGrad:output:0*
T0*(
_output_shapes
: ôô*
paddingVALID*
strides
£
4gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhtc/Cast:y:0/gradient_tape/htc/conv2d/Conv2D/ShapeN:output:1<gradient_tape/htc/max_pooling2d/MaxPool/MaxPoolGrad:output:0*
T0*&
_output_shapes
: *
paddingVALID*
strides

IdentityIdentity=gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
: r

Identity_1Identity5gradient_tape/htc/conv2d/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
: 

Identity_2IdentityKgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes
: 

Identity_3IdentityLgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes
: 

Identity_4Identity?gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
: @t

Identity_5Identity7gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:@

Identity_6IdentityMgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes
:@

Identity_7IdentityNgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes
:@

Identity_8Identity?gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilter:output:0*
T0*'
_output_shapes
:@u

Identity_9Identity7gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:
Identity_10IdentityMgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes	
:
Identity_11IdentityNgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes	
:
Identity_12Identity?gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilter:output:0*
T0*(
_output_shapes
:v
Identity_13Identity7gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:
Identity_14IdentityMgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes	
:
Identity_15IdentityNgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes	
:
Identity_16Identity?gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilter:output:0*
T0*(
_output_shapes
:v
Identity_17Identity7gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:
Identity_18IdentityMgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:scale_backprop:0*
T0*
_output_shapes	
:
Identity_19IdentityNgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:offset_backprop:0*
T0*
_output_shapes	
:t
Identity_20Identity1gradient_tape/htc/dense/MatMul/MatMul_1:product:0*
T0*
_output_shapes
:	$r
Identity_21Identity4gradient_tape/htc/dense/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:}
Identity_22Identity?gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1:z:0*
T0*
_output_shapes
:
Identity_23IdentityHgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0*
T0*
_output_shapes
:¨
	IdentityN	IdentityN=gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilter:output:05gradient_tape/htc/conv2d/BiasAdd/BiasAddGrad:output:0Kgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:scale_backprop:0Lgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:offset_backprop:01gradient_tape/htc/dense/MatMul/MatMul_1:product:04gradient_tape/htc/dense/BiasAdd/BiasAddGrad:output:0?gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1:z:0Hgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0=gradient_tape/htc/conv2d/Conv2D/Conv2DBackpropFilter:output:05gradient_tape/htc/conv2d/BiasAdd/BiasAddGrad:output:0Kgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:scale_backprop:0Lgradient_tape/htc/batch_normalization/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_1/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_1/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_1/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_2/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_2/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_2/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_3/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_3/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_3/FusedBatchNormGradV3:offset_backprop:0?gradient_tape/htc/conv2d_4/Conv2D/Conv2DBackpropFilter:output:07gradient_tape/htc/conv2d_4/BiasAdd/BiasAddGrad:output:0Mgradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:scale_backprop:0Ngradient_tape/htc/batch_normalization_4/FusedBatchNormGradV3:offset_backprop:01gradient_tape/htc/dense/MatMul/MatMul_1:product:04gradient_tape/htc/dense/BiasAdd/BiasAddGrad:output:0?gradient_tape/htc/batch_normalization_5/batchnorm/mul/Mul_1:z:0Hgradient_tape/htc/batch_normalization_5/batchnorm/add_1/Reshape:output:0*9
T4
220*,
_gradient_op_typeCustomGradient-296136*Ô
_output_shapesÁ
¾: : : : : @:@:@:@:@::::::::::::	$:::: : : : : @:@:@:@:@::::::::::::	$:::±
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296248µ
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296297¿
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296344Ã
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296391¸
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296438»
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296485Å
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296532É
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296579¹
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296626»
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296673Ç
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296720Ë
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296767»
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296814½
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296861Ç
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296908Ë
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_296955»
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297002½
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297049Ç
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297096Ë
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297143µ
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297190·
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297237Õ
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297284Ñ
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
GPU2 *0J 8 *,
f'R%
#__inference__update_step_xla_297331G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
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
: 
SumSum7sparse_categorical_crossentropy/weighted_loss/value:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: ¥
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
: ¹
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceCast:y:0^AssignAddVariableOp_1*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0 
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1^AssignAddVariableOp_2*
_output_shapes
: *
dtype0
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
ÿÿÿÿÿÿÿÿÿr
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
: §
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
: »
AssignAddVariableOp_4AssignAddVariableOpassignaddvariableop_4_resource
Cast_4:y:0^AssignAddVariableOp_3*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0¢
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3^AssignAddVariableOp_4*
_output_shapes
: *
dtype0
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_4_resource^AssignAddVariableOp_4*
_output_shapes
: *
dtype0
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: J
Identity_25Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: *(
_construction_contextkEagerRuntime*ã
_input_shapesÑ
Î: ôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
: ôô
 
_user_specified_nameimages:B>

_output_shapes
: 
 
_user_specified_namelabels
½
L
0__inference_max_pooling2d_2_layer_call_fn_298140

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_294295
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³


D__inference_conv2d_3_layer_call_and_return_conditional_losses_294712

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
D
(__inference_re_lu_1_layer_call_fn_298111

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_294667h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È 
±
#__inference__update_step_xla_296391
gradient
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource: +
sub_3_readvariableop_resource: ¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: 
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
: 
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
 *¿Ö3Q
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
Ä
D
(__inference_re_lu_2_layer_call_fn_298212

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_294700i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
J
.__inference_max_pooling2d_layer_call_fn_297938

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_294143
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

O__inference_batch_normalization_layer_call_and_return_conditional_losses_297987

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È 
±
#__inference__update_step_xla_296485
gradient
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:@+
sub_3_readvariableop_resource:@¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:@
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:@
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:@*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:@
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
 *¿Ö3Q
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

¾
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294199

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢!
À
#__inference__update_step_xla_297190
gradient
variable:	$!
readvariableop_resource:	 #
readvariableop_1_resource: 0
sub_2_readvariableop_resource:	$0
sub_3_readvariableop_resource:	$¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:	$*
dtype0^
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	$L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=S
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:	$
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0D
SquareSquaregradient*
T0*
_output_shapes
:	$s
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
:	$*
dtype0`
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	$L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:S
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:	$
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:	$*
dtype0]
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:	$
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes
:	$*
dtype0W
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	$L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3V
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes
:	$T
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes
:	$f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:	$: : : : : *
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
:	$
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

È
$__inference_htc_layer_call_fn_297603
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

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
 #$*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_htc_layer_call_and_return_conditional_losses_295205o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô

_user_specified_namex
Ü
 
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298391

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_298318

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­t

?__inference_htc_layer_call_and_return_conditional_losses_294821
x'
conv2d_294614: 
conv2d_294616: (
batch_normalization_294620: (
batch_normalization_294622: (
batch_normalization_294624: (
batch_normalization_294626: )
conv2d_1_294647: @
conv2d_1_294649:@*
batch_normalization_1_294653:@*
batch_normalization_1_294655:@*
batch_normalization_1_294657:@*
batch_normalization_1_294659:@*
conv2d_2_294680:@
conv2d_2_294682:	+
batch_normalization_2_294686:	+
batch_normalization_2_294688:	+
batch_normalization_2_294690:	+
batch_normalization_2_294692:	+
conv2d_3_294713:
conv2d_3_294715:	+
batch_normalization_3_294719:	+
batch_normalization_3_294721:	+
batch_normalization_3_294723:	+
batch_normalization_3_294725:	+
conv2d_4_294746:
conv2d_4_294748:	+
batch_normalization_4_294752:	+
batch_normalization_4_294754:	+
batch_normalization_4_294756:	+
batch_normalization_4_294758:	
dense_294799:	$
dense_294801:*
batch_normalization_5_294804:*
batch_normalization_5_294806:*
batch_normalization_5_294808:*
batch_normalization_5_294810:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCallò
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_294614conv2d_294616*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_294613ð
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_294143
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_294620batch_normalization_294622batch_normalization_294624batch_normalization_294626*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294168í
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_294634
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_294647conv2d_1_294649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_294646ö
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_294219
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_294653batch_normalization_1_294655batch_normalization_1_294657batch_normalization_1_294659*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294244ó
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_294667
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_294680conv2d_2_294682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_294679÷
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_294295
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_294686batch_normalization_2_294688batch_normalization_2_294690batch_normalization_2_294692*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294320ô
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_294700
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_294713conv2d_3_294715*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_294712÷
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_294371
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_294719batch_normalization_3_294721batch_normalization_3_294723batch_normalization_3_294725*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294396ô
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_294733
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_294746conv2d_4_294748*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_294745÷
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_294447
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_294752batch_normalization_4_294754batch_normalization_4_294756batch_normalization_4_294758*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294472ô
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_294766g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
dropout/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_294775i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
transpose_1	Transpose dropout/PartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_294785
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_294799dense_294801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_294798
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_294804batch_normalization_5_294806batch_normalization_5_294808batch_normalization_5_294810*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294538ñ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_294818r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿôô

_user_specified_namex
È 
±
#__inference__update_step_xla_297331
gradient
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:+
sub_3_readvariableop_resource:¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:
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
 *¿Ö3Q
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
²$
Ò
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294585

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOph
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

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
 *o:q
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
:ÿÿÿÿÿÿÿÿÿh
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
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
D
(__inference_flatten_layer_call_fn_298451

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_294785a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
_
C__inference_flatten_layer_call_and_return_conditional_losses_298457

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

û
B__inference_conv2d_layer_call_and_return_conditional_losses_294613

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *
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
:ÿÿÿÿÿÿÿÿÿíí i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ë
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_294700

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²"
Ø
#__inference__update_step_xla_296626
gradient#
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: 8
sub_2_readvariableop_resource:@8
sub_3_readvariableop_resource:@¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:@*
dtype0f
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=[
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*'
_output_shapes
:@
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0L
SquareSquaregradient*
T0*'
_output_shapes
:@{
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*'
_output_shapes
:@*
dtype0h
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:@L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:[
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*'
_output_shapes
:@
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*'
_output_shapes
:@*
dtype0e
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*'
_output_shapes
:@
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*'
_output_shapes
:@*
dtype0_
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3^
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*'
_output_shapes
:@\
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*'
_output_shapes
:@f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*0
_input_shapes
:@: : : : : *
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
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_294295

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_2_layer_call_fn_298171

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294351
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä"
Û
#__inference__update_step_xla_297002
gradient$
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: 9
sub_2_readvariableop_resource:9
sub_3_readvariableop_resource:¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0g
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=\
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*(
_output_shapes
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0M
SquareSquaregradient*
T0*(
_output_shapes
:|
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*(
_output_shapes
:*
dtype0i
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:\
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*(
_output_shapes
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*(
_output_shapes
:*
dtype0f
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*(
_output_shapes
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*(
_output_shapes
:*
dtype0`
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3_
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*(
_output_shapes
:]
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*(
_output_shapes
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*1
_input_shapes 
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

Ä
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294503

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
 
)__inference_conv2d_2_layer_call_fn_298125

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_294679x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú 
´
#__inference__update_step_xla_297096
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ë
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_298217

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Óu
Ã
?__inference_htc_layer_call_and_return_conditional_losses_295567
input_1'
conv2d_295465: 
conv2d_295467: (
batch_normalization_295471: (
batch_normalization_295473: (
batch_normalization_295475: (
batch_normalization_295477: )
conv2d_1_295481: @
conv2d_1_295483:@*
batch_normalization_1_295487:@*
batch_normalization_1_295489:@*
batch_normalization_1_295491:@*
batch_normalization_1_295493:@*
conv2d_2_295497:@
conv2d_2_295499:	+
batch_normalization_2_295503:	+
batch_normalization_2_295505:	+
batch_normalization_2_295507:	+
batch_normalization_2_295509:	+
conv2d_3_295513:
conv2d_3_295515:	+
batch_normalization_3_295519:	+
batch_normalization_3_295521:	+
batch_normalization_3_295523:	+
batch_normalization_3_295525:	+
conv2d_4_295529:
conv2d_4_295531:	+
batch_normalization_4_295535:	+
batch_normalization_4_295537:	+
batch_normalization_4_295539:	+
batch_normalization_4_295541:	
dense_295551:	$
dense_295553:*
batch_normalization_5_295556:*
batch_normalization_5_295558:*
batch_normalization_5_295560:*
batch_normalization_5_295562:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCallø
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_295465conv2d_295467*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_294613ð
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_294143
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_295471batch_normalization_295473batch_normalization_295475batch_normalization_295477*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294199í
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_294634
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_295481conv2d_1_295483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_294646ö
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_294219
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_295487batch_normalization_1_295489batch_normalization_1_295491batch_normalization_1_295493*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294275ó
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_294667
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_295497conv2d_2_295499*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_294679÷
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_294295
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_295503batch_normalization_2_295505batch_normalization_2_295507batch_normalization_2_295509*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294351ô
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_294700
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_295513conv2d_3_295515*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_294712÷
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_294371
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_295519batch_normalization_3_295521batch_normalization_3_295523batch_normalization_3_295525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294427ô
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_294733
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_295529conv2d_4_295531*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_294745÷
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_294447
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_295535batch_normalization_4_295537batch_normalization_4_295539batch_normalization_4_295541*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294503ô
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_294766g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
dropout/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_294938i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
transpose_1	Transpose(dropout/StatefulPartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_294785
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_295551dense_295553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_294798
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_295556batch_normalization_5_295558batch_normalization_5_295560batch_normalization_5_295562*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294585ñ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_294818r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿôô
!
_user_specified_name	input_1
÷
Ì#
?__inference_htc_layer_call_and_return_conditional_losses_297914
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
'conv2d_2_conv2d_readvariableop_resource:@7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_4_conv2d_readvariableop_resource:7
(conv2d_4_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	7
$dense_matmul_readvariableop_resource:	$3
%dense_biasadd_readvariableop_resource:K
=batch_normalization_5_assignmovingavg_readvariableop_resource:M
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_5_cast_readvariableop_resource:B
4batch_normalization_5_cast_1_readvariableop_resource:
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1¢5batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_4/ReadVariableOp¢&batch_normalization_4/ReadVariableOp_1¢%batch_normalization_5/AssignMovingAvg¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢'batch_normalization_5/AssignMovingAvg_1¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_5/Cast/ReadVariableOp¢+batch_normalization_5/Cast_1/ReadVariableOp¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¥
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí ¦
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
ksize
*
paddingVALID*
strides

"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0À
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿbb : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape( 
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¾
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@ª
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Á
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ñ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Á
conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ñ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Á
conv2d_4/Conv2DConv2Dre_lu_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ñ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transposere_lu_4/Relu:activations:0transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout/MulMultranspose:y:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/dropout/ShapeShapetranspose:y:0*
T0*
_output_shapes
:¥
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ç
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ¼
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
transpose_1	Transpose!dropout/dropout/SelectV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   v
flatten/ReshapeReshapetranspose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	$*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¼
"batch_normalization_5/moments/meanMeandense/Softmax:softmax:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:Ä
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense/Softmax:softmax:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: à
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
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
×#<®
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Ã
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:º
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
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
×#<²
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
:À
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes
:*
dtype0
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:¯
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 
%batch_normalization_5/batchnorm/mul_1Muldense/Softmax:softmax:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:­
#batch_normalization_5/batchnorm/subSub1batch_normalization_5/Cast/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:´
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:ÿÿÿÿÿÿÿÿÿôô

_user_specified_namex
¦
G
+__inference_activation_layer_call_fn_298562

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_294818`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²$
Ò
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298557

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOph
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

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
 *o:q
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
:ÿÿÿÿÿÿÿÿÿh
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
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 "
Õ
#__inference__update_step_xla_296438
gradient"
variable: @!
readvariableop_resource:	 #
readvariableop_1_resource: 7
sub_2_readvariableop_resource: @7
sub_3_readvariableop_resource: @¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=Z
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*&
_output_shapes
: @
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
 *o:Z
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*&
_output_shapes
: @
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*&
_output_shapes
: @*
dtype0d
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*&
_output_shapes
: @
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
 *¿Ö3]
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
È

b
C__inference_dropout_layer_call_and_return_conditional_losses_294938

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294427

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_294667

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_294219

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
Î
$__inference_htc_layer_call_fn_294896
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

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall­
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
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_htc_layer_call_and_return_conditional_losses_294821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
!
_user_specified_name	input_1
È 
±
#__inference__update_step_xla_297237
gradient
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:+
sub_3_readvariableop_resource:¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:
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
 *¿Ö3Q
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
	
Ñ
6__inference_batch_normalization_1_layer_call_fn_298070

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294275
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ¬
-
__inference__traced_save_299020
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

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ª#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*Ó"
valueÉ"BÆ"_B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*Ó
valueÉBÆ_B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B î+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop,savev2_htc_conv2d_kernel_read_readvariableop*savev2_htc_conv2d_bias_read_readvariableop8savev2_htc_batch_normalization_gamma_read_readvariableop7savev2_htc_batch_normalization_beta_read_readvariableop>savev2_htc_batch_normalization_moving_mean_read_readvariableopBsavev2_htc_batch_normalization_moving_variance_read_readvariableop.savev2_htc_conv2d_1_kernel_read_readvariableop,savev2_htc_conv2d_1_bias_read_readvariableop:savev2_htc_batch_normalization_1_gamma_read_readvariableop9savev2_htc_batch_normalization_1_beta_read_readvariableop@savev2_htc_batch_normalization_1_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_1_moving_variance_read_readvariableop.savev2_htc_conv2d_2_kernel_read_readvariableop,savev2_htc_conv2d_2_bias_read_readvariableop:savev2_htc_batch_normalization_2_gamma_read_readvariableop9savev2_htc_batch_normalization_2_beta_read_readvariableop@savev2_htc_batch_normalization_2_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_2_moving_variance_read_readvariableop.savev2_htc_conv2d_3_kernel_read_readvariableop,savev2_htc_conv2d_3_bias_read_readvariableop:savev2_htc_batch_normalization_3_gamma_read_readvariableop9savev2_htc_batch_normalization_3_beta_read_readvariableop@savev2_htc_batch_normalization_3_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_3_moving_variance_read_readvariableop.savev2_htc_conv2d_4_kernel_read_readvariableop,savev2_htc_conv2d_4_bias_read_readvariableop:savev2_htc_batch_normalization_4_gamma_read_readvariableop9savev2_htc_batch_normalization_4_beta_read_readvariableop@savev2_htc_batch_normalization_4_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_4_moving_variance_read_readvariableop+savev2_htc_dense_kernel_read_readvariableop)savev2_htc_dense_bias_read_readvariableop:savev2_htc_batch_normalization_5_gamma_read_readvariableop9savev2_htc_batch_normalization_5_beta_read_readvariableop@savev2_htc_batch_normalization_5_moving_mean_read_readvariableopDsavev2_htc_batch_normalization_5_moving_variance_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop3savev2_adam_m_htc_conv2d_kernel_read_readvariableop3savev2_adam_v_htc_conv2d_kernel_read_readvariableop1savev2_adam_m_htc_conv2d_bias_read_readvariableop1savev2_adam_v_htc_conv2d_bias_read_readvariableop?savev2_adam_m_htc_batch_normalization_gamma_read_readvariableop?savev2_adam_v_htc_batch_normalization_gamma_read_readvariableop>savev2_adam_m_htc_batch_normalization_beta_read_readvariableop>savev2_adam_v_htc_batch_normalization_beta_read_readvariableop5savev2_adam_m_htc_conv2d_1_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_1_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_1_bias_read_readvariableop3savev2_adam_v_htc_conv2d_1_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_1_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_1_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_1_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_1_beta_read_readvariableop5savev2_adam_m_htc_conv2d_2_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_2_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_2_bias_read_readvariableop3savev2_adam_v_htc_conv2d_2_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_2_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_2_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_2_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_2_beta_read_readvariableop5savev2_adam_m_htc_conv2d_3_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_3_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_3_bias_read_readvariableop3savev2_adam_v_htc_conv2d_3_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_3_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_3_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_3_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_3_beta_read_readvariableop5savev2_adam_m_htc_conv2d_4_kernel_read_readvariableop5savev2_adam_v_htc_conv2d_4_kernel_read_readvariableop3savev2_adam_m_htc_conv2d_4_bias_read_readvariableop3savev2_adam_v_htc_conv2d_4_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_4_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_4_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_4_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_4_beta_read_readvariableop2savev2_adam_m_htc_dense_kernel_read_readvariableop2savev2_adam_v_htc_dense_kernel_read_readvariableop0savev2_adam_m_htc_dense_bias_read_readvariableop0savev2_adam_v_htc_dense_bias_read_readvariableopAsavev2_adam_m_htc_batch_normalization_5_gamma_read_readvariableopAsavev2_adam_v_htc_batch_normalization_5_gamma_read_readvariableop@savev2_adam_m_htc_batch_normalization_5_beta_read_readvariableop@savev2_adam_v_htc_batch_normalization_5_beta_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *m
dtypesc
a2_	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
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

identity_1Identity_1:output:0*
_input_shapes
: : : : : : : : : : : : : : : : @:@:@:@:@:@:@::::::::::::::::::	$:::::: : : : : : : : : : : @: @:@:@:@:@:@:@:@:@:::::::::::::::::::::::	$:	$::::::: 2(
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
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::.!*
(
_output_shapes
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::%'!

_output_shapes
:	$: (
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
:@:-@)
'
_output_shapes
:@:!A

_output_shapes	
::!B

_output_shapes	
::!C

_output_shapes	
::!D

_output_shapes	
::!E

_output_shapes	
::!F

_output_shapes	
::.G*
(
_output_shapes
::.H*
(
_output_shapes
::!I

_output_shapes	
::!J

_output_shapes	
::!K

_output_shapes	
::!L

_output_shapes	
::!M

_output_shapes	
::!N

_output_shapes	
::.O*
(
_output_shapes
::.P*
(
_output_shapes
::!Q

_output_shapes	
::!R

_output_shapes	
::!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::!V

_output_shapes	
::%W!

_output_shapes
:	$:%X!

_output_shapes
:	$: Y
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

¾
O__inference_batch_normalization_layer_call_and_return_conditional_losses_298005

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298308

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298189

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Áu
½
?__inference_htc_layer_call_and_return_conditional_losses_295205
x'
conv2d_295103: 
conv2d_295105: (
batch_normalization_295109: (
batch_normalization_295111: (
batch_normalization_295113: (
batch_normalization_295115: )
conv2d_1_295119: @
conv2d_1_295121:@*
batch_normalization_1_295125:@*
batch_normalization_1_295127:@*
batch_normalization_1_295129:@*
batch_normalization_1_295131:@*
conv2d_2_295135:@
conv2d_2_295137:	+
batch_normalization_2_295141:	+
batch_normalization_2_295143:	+
batch_normalization_2_295145:	+
batch_normalization_2_295147:	+
conv2d_3_295151:
conv2d_3_295153:	+
batch_normalization_3_295157:	+
batch_normalization_3_295159:	+
batch_normalization_3_295161:	+
batch_normalization_3_295163:	+
conv2d_4_295167:
conv2d_4_295169:	+
batch_normalization_4_295173:	+
batch_normalization_4_295175:	+
batch_normalization_4_295177:	+
batch_normalization_4_295179:	
dense_295189:	$
dense_295191:*
batch_normalization_5_295194:*
batch_normalization_5_295196:*
batch_normalization_5_295198:*
batch_normalization_5_295200:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCallò
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_295103conv2d_295105*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_294613ð
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_294143
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_295109batch_normalization_295111batch_normalization_295113batch_normalization_295115*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294199í
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_294634
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_295119conv2d_1_295121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_294646ö
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_294219
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_295125batch_normalization_1_295127batch_normalization_1_295129batch_normalization_1_295131*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294275ó
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_294667
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_295135conv2d_2_295137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_294679÷
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_294295
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_295141batch_normalization_2_295143batch_normalization_2_295145batch_normalization_2_295147*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294351ô
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_294700
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_295151conv2d_3_295153*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_294712÷
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_294371
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_295157batch_normalization_3_295159batch_normalization_3_295161batch_normalization_3_295163*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294427ô
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_294733
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_295167conv2d_4_295169*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_294745÷
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_294447
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_295173batch_normalization_4_295175batch_normalization_4_295177batch_normalization_4_295179*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294503ô
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_294766g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
dropout/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_294938i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
transpose_1	Transpose(dropout/StatefulPartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_294785
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_295189dense_295191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_294798
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_295194batch_normalization_5_295196batch_normalization_5_295198batch_normalization_5_295200*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294585ñ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_294818r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿôô

_user_specified_namex
ç
_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_298116

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³


D__inference_conv2d_4_layer_call_and_return_conditional_losses_298337

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

ÿ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_298135

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿t
¡
?__inference_htc_layer_call_and_return_conditional_losses_295462
input_1'
conv2d_295360: 
conv2d_295362: (
batch_normalization_295366: (
batch_normalization_295368: (
batch_normalization_295370: (
batch_normalization_295372: )
conv2d_1_295376: @
conv2d_1_295378:@*
batch_normalization_1_295382:@*
batch_normalization_1_295384:@*
batch_normalization_1_295386:@*
batch_normalization_1_295388:@*
conv2d_2_295392:@
conv2d_2_295394:	+
batch_normalization_2_295398:	+
batch_normalization_2_295400:	+
batch_normalization_2_295402:	+
batch_normalization_2_295404:	+
conv2d_3_295408:
conv2d_3_295410:	+
batch_normalization_3_295414:	+
batch_normalization_3_295416:	+
batch_normalization_3_295418:	+
batch_normalization_3_295420:	+
conv2d_4_295424:
conv2d_4_295426:	+
batch_normalization_4_295430:	+
batch_normalization_4_295432:	+
batch_normalization_4_295434:	+
batch_normalization_4_295436:	
dense_295446:	$
dense_295448:*
batch_normalization_5_295451:*
batch_normalization_5_295453:*
batch_normalization_5_295455:*
batch_normalization_5_295457:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCallø
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_295360conv2d_295362*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_294613ð
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_294143
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_295366batch_normalization_295368batch_normalization_295370batch_normalization_295372*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294168í
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_294634
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_295376conv2d_1_295378*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_294646ö
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_294219
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_295382batch_normalization_1_295384batch_normalization_1_295386batch_normalization_1_295388*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294244ó
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_294667
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_295392conv2d_2_295394*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_294679÷
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_294295
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_295398batch_normalization_2_295400batch_normalization_2_295402batch_normalization_2_295404*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294320ô
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_294700
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_3_295408conv2d_3_295410*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_294712÷
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_294371
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_295414batch_normalization_3_295416batch_normalization_3_295418batch_normalization_3_295420*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294396ô
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_294733
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_4_295424conv2d_4_295426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_294745÷
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_294447
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_4_295430batch_normalization_4_295432batch_normalization_4_295434batch_normalization_4_295436*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294472ô
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_294766g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transpose re_lu_4/PartitionedCall:output:0transpose/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
dropout/PartitionedCallPartitionedCalltranspose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_294775i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             
transpose_1	Transpose dropout/PartitionedCall:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
flatten/PartitionedCallPartitionedCalltranspose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_294785
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_295446dense_295448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_294798
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_5_295451batch_normalization_5_295453batch_normalization_5_295455batch_normalization_5_295457*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294538ñ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_294818r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿôô
!
_user_specified_name	input_1

È
$__inference_htc_layer_call_fn_297526
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

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall§
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
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_htc_layer_call_and_return_conditional_losses_294821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô

_user_specified_namex
°

û
B__inference_conv2d_layer_call_and_return_conditional_losses_297933

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí *
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
:ÿÿÿÿÿÿÿÿÿíí i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿíí w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
­
Ñ
6__inference_batch_normalization_5_layer_call_fn_298490

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_3_layer_call_fn_298259

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294396
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
$__inference_htc_layer_call_fn_295357
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

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¡
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
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
 #$*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_htc_layer_call_and_return_conditional_losses_295205o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
!
_user_specified_name	input_1
	
Ñ
6__inference_batch_normalization_1_layer_call_fn_298057

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294244
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú 
´
#__inference__update_step_xla_296767
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
È

b
C__inference_dropout_layer_call_and_return_conditional_losses_298446

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È 
±
#__inference__update_step_xla_296532
gradient
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:@+
sub_3_readvariableop_resource:@¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:@
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:@
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:@*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:@
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
 *¿Ö3Q
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
Ü
 
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298290

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294351

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú 
´
#__inference__update_step_xla_296673
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

À
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_298106

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¡

ó
A__inference_dense_layer_call_and_return_conditional_losses_294798

inputs1
matmul_readvariableop_resource:	$-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_4_layer_call_fn_298360

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294472
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú 
´
#__inference__update_step_xla_297143
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
È 
±
#__inference__update_step_xla_296297
gradient
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource: +
sub_3_readvariableop_resource: ¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: 
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
: 
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
 *¿Ö3Q
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
È 
±
#__inference__update_step_xla_297284
gradient
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:+
sub_3_readvariableop_resource:¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:
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
 *¿Ö3Q
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

Ú@
"__inference__traced_restore_299312
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
'assignvariableop_20_htc_conv2d_2_kernel:@4
%assignvariableop_21_htc_conv2d_2_bias:	B
3assignvariableop_22_htc_batch_normalization_2_gamma:	A
2assignvariableop_23_htc_batch_normalization_2_beta:	H
9assignvariableop_24_htc_batch_normalization_2_moving_mean:	L
=assignvariableop_25_htc_batch_normalization_2_moving_variance:	C
'assignvariableop_26_htc_conv2d_3_kernel:4
%assignvariableop_27_htc_conv2d_3_bias:	B
3assignvariableop_28_htc_batch_normalization_3_gamma:	A
2assignvariableop_29_htc_batch_normalization_3_beta:	H
9assignvariableop_30_htc_batch_normalization_3_moving_mean:	L
=assignvariableop_31_htc_batch_normalization_3_moving_variance:	C
'assignvariableop_32_htc_conv2d_4_kernel:4
%assignvariableop_33_htc_conv2d_4_bias:	B
3assignvariableop_34_htc_batch_normalization_4_gamma:	A
2assignvariableop_35_htc_batch_normalization_4_beta:	H
9assignvariableop_36_htc_batch_normalization_4_moving_mean:	L
=assignvariableop_37_htc_batch_normalization_4_moving_variance:	7
$assignvariableop_38_htc_dense_kernel:	$0
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
.assignvariableop_62_adam_m_htc_conv2d_2_kernel:@I
.assignvariableop_63_adam_v_htc_conv2d_2_kernel:@;
,assignvariableop_64_adam_m_htc_conv2d_2_bias:	;
,assignvariableop_65_adam_v_htc_conv2d_2_bias:	I
:assignvariableop_66_adam_m_htc_batch_normalization_2_gamma:	I
:assignvariableop_67_adam_v_htc_batch_normalization_2_gamma:	H
9assignvariableop_68_adam_m_htc_batch_normalization_2_beta:	H
9assignvariableop_69_adam_v_htc_batch_normalization_2_beta:	J
.assignvariableop_70_adam_m_htc_conv2d_3_kernel:J
.assignvariableop_71_adam_v_htc_conv2d_3_kernel:;
,assignvariableop_72_adam_m_htc_conv2d_3_bias:	;
,assignvariableop_73_adam_v_htc_conv2d_3_bias:	I
:assignvariableop_74_adam_m_htc_batch_normalization_3_gamma:	I
:assignvariableop_75_adam_v_htc_batch_normalization_3_gamma:	H
9assignvariableop_76_adam_m_htc_batch_normalization_3_beta:	H
9assignvariableop_77_adam_v_htc_batch_normalization_3_beta:	J
.assignvariableop_78_adam_m_htc_conv2d_4_kernel:J
.assignvariableop_79_adam_v_htc_conv2d_4_kernel:;
,assignvariableop_80_adam_m_htc_conv2d_4_bias:	;
,assignvariableop_81_adam_v_htc_conv2d_4_bias:	I
:assignvariableop_82_adam_m_htc_batch_normalization_4_gamma:	I
:assignvariableop_83_adam_v_htc_batch_normalization_4_gamma:	H
9assignvariableop_84_adam_m_htc_batch_normalization_4_beta:	H
9assignvariableop_85_adam_v_htc_batch_normalization_4_beta:	>
+assignvariableop_86_adam_m_htc_dense_kernel:	$>
+assignvariableop_87_adam_v_htc_dense_kernel:	$7
)assignvariableop_88_adam_m_htc_dense_bias:7
)assignvariableop_89_adam_v_htc_dense_bias:H
:assignvariableop_90_adam_m_htc_batch_normalization_5_gamma:H
:assignvariableop_91_adam_v_htc_batch_normalization_5_gamma:G
9assignvariableop_92_adam_m_htc_batch_normalization_5_beta:G
9assignvariableop_93_adam_v_htc_batch_normalization_5_beta:
identity_95¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93­#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*Ó"
valueÉ"BÆ"_B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*Ó
valueÉBÆ_B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ü
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesÿ
ü:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*m
dtypesc
a2_	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOpAssignVariableOpassignvariableop_total_3Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_1AssignVariableOpassignvariableop_1_count_3Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_2AssignVariableOpassignvariableop_2_total_2Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_3AssignVariableOpassignvariableop_3_count_2Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_4AssignVariableOpassignvariableop_4_total_1Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_8AssignVariableOp$assignvariableop_8_htc_conv2d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_9AssignVariableOp"assignvariableop_9_htc_conv2d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_10AssignVariableOp1assignvariableop_10_htc_batch_normalization_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_11AssignVariableOp0assignvariableop_11_htc_batch_normalization_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_12AssignVariableOp7assignvariableop_12_htc_batch_normalization_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ô
AssignVariableOp_13AssignVariableOp;assignvariableop_13_htc_batch_normalization_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_14AssignVariableOp'assignvariableop_14_htc_conv2d_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_15AssignVariableOp%assignvariableop_15_htc_conv2d_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_16AssignVariableOp3assignvariableop_16_htc_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_17AssignVariableOp2assignvariableop_17_htc_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_18AssignVariableOp9assignvariableop_18_htc_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_19AssignVariableOp=assignvariableop_19_htc_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_20AssignVariableOp'assignvariableop_20_htc_conv2d_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_21AssignVariableOp%assignvariableop_21_htc_conv2d_2_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_22AssignVariableOp3assignvariableop_22_htc_batch_normalization_2_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_23AssignVariableOp2assignvariableop_23_htc_batch_normalization_2_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_24AssignVariableOp9assignvariableop_24_htc_batch_normalization_2_moving_meanIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_25AssignVariableOp=assignvariableop_25_htc_batch_normalization_2_moving_varianceIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_26AssignVariableOp'assignvariableop_26_htc_conv2d_3_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_27AssignVariableOp%assignvariableop_27_htc_conv2d_3_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_28AssignVariableOp3assignvariableop_28_htc_batch_normalization_3_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_29AssignVariableOp2assignvariableop_29_htc_batch_normalization_3_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_30AssignVariableOp9assignvariableop_30_htc_batch_normalization_3_moving_meanIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_31AssignVariableOp=assignvariableop_31_htc_batch_normalization_3_moving_varianceIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_32AssignVariableOp'assignvariableop_32_htc_conv2d_4_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_33AssignVariableOp%assignvariableop_33_htc_conv2d_4_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_34AssignVariableOp3assignvariableop_34_htc_batch_normalization_4_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_35AssignVariableOp2assignvariableop_35_htc_batch_normalization_4_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_36AssignVariableOp9assignvariableop_36_htc_batch_normalization_4_moving_meanIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_37AssignVariableOp=assignvariableop_37_htc_batch_normalization_4_moving_varianceIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_38AssignVariableOp$assignvariableop_38_htc_dense_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_39AssignVariableOp"assignvariableop_39_htc_dense_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_40AssignVariableOp3assignvariableop_40_htc_batch_normalization_5_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_41AssignVariableOp2assignvariableop_41_htc_batch_normalization_5_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_42AssignVariableOp9assignvariableop_42_htc_batch_normalization_5_moving_meanIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_43AssignVariableOp=assignvariableop_43_htc_batch_normalization_5_moving_varianceIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0	*
_output_shapes
:¶
AssignVariableOp_44AssignVariableOpassignvariableop_44_iterationIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_45AssignVariableOp!assignvariableop_45_learning_rateIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_m_htc_conv2d_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_v_htc_conv2d_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_htc_conv2d_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_htc_conv2d_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adam_m_htc_batch_normalization_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_51AssignVariableOp8assignvariableop_51_adam_v_htc_batch_normalization_gammaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_m_htc_batch_normalization_betaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_v_htc_batch_normalization_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_54AssignVariableOp.assignvariableop_54_adam_m_htc_conv2d_1_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_55AssignVariableOp.assignvariableop_55_adam_v_htc_conv2d_1_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_m_htc_conv2d_1_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_v_htc_conv2d_1_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_58AssignVariableOp:assignvariableop_58_adam_m_htc_batch_normalization_1_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_59AssignVariableOp:assignvariableop_59_adam_v_htc_batch_normalization_1_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_60AssignVariableOp9assignvariableop_60_adam_m_htc_batch_normalization_1_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_61AssignVariableOp9assignvariableop_61_adam_v_htc_batch_normalization_1_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_62AssignVariableOp.assignvariableop_62_adam_m_htc_conv2d_2_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_63AssignVariableOp.assignvariableop_63_adam_v_htc_conv2d_2_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_m_htc_conv2d_2_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_v_htc_conv2d_2_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_66AssignVariableOp:assignvariableop_66_adam_m_htc_batch_normalization_2_gammaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_67AssignVariableOp:assignvariableop_67_adam_v_htc_batch_normalization_2_gammaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_68AssignVariableOp9assignvariableop_68_adam_m_htc_batch_normalization_2_betaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_69AssignVariableOp9assignvariableop_69_adam_v_htc_batch_normalization_2_betaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_70AssignVariableOp.assignvariableop_70_adam_m_htc_conv2d_3_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_v_htc_conv2d_3_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_72AssignVariableOp,assignvariableop_72_adam_m_htc_conv2d_3_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adam_v_htc_conv2d_3_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_74AssignVariableOp:assignvariableop_74_adam_m_htc_batch_normalization_3_gammaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_75AssignVariableOp:assignvariableop_75_adam_v_htc_batch_normalization_3_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_76AssignVariableOp9assignvariableop_76_adam_m_htc_batch_normalization_3_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_77AssignVariableOp9assignvariableop_77_adam_v_htc_batch_normalization_3_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_78AssignVariableOp.assignvariableop_78_adam_m_htc_conv2d_4_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_v_htc_conv2d_4_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_m_htc_conv2d_4_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_v_htc_conv2d_4_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_82AssignVariableOp:assignvariableop_82_adam_m_htc_batch_normalization_4_gammaIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_83AssignVariableOp:assignvariableop_83_adam_v_htc_batch_normalization_4_gammaIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_84AssignVariableOp9assignvariableop_84_adam_m_htc_batch_normalization_4_betaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_85AssignVariableOp9assignvariableop_85_adam_v_htc_batch_normalization_4_betaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adam_m_htc_dense_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_v_htc_dense_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_m_htc_dense_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_89AssignVariableOp)assignvariableop_89_adam_v_htc_dense_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_90AssignVariableOp:assignvariableop_90_adam_m_htc_batch_normalization_5_gammaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_91AssignVariableOp:assignvariableop_91_adam_v_htc_batch_normalization_5_gammaIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_92AssignVariableOp9assignvariableop_92_adam_m_htc_batch_normalization_5_betaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:Ò
AssignVariableOp_93AssignVariableOp9assignvariableop_93_adam_v_htc_batch_normalization_5_betaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ã
Identity_94Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_95IdentityIdentity_94:output:0^NoOp_1*
T0*
_output_shapes
: Ð
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93*"
_acd_function_control_output(*
_output_shapes
 "#
identity_95Identity_95:output:0*Ó
_input_shapesÁ
¾: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
¨

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_298034

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*
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
:ÿÿÿÿÿÿÿÿÿ^^@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿbb : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
ë
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_294766

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298523

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
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
 *o:t
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
:ÿÿÿÿÿÿÿÿÿk
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
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
]
A__inference_re_lu_layer_call_and_return_conditional_losses_298015

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿbb :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298207

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
b
F__inference_activation_layer_call_and_return_conditional_losses_298567

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È 
±
#__inference__update_step_xla_296579
gradient
variable:@!
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource:@+
sub_3_readvariableop_resource:@¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
:@
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
:@
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
:@*
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
:@
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
 *¿Ö3Q
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
Ê

O__inference_batch_normalization_layer_call_and_return_conditional_losses_294168

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294396

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_3_layer_call_fn_298272

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_294427
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
a
C__inference_dropout_layer_call_and_return_conditional_losses_294775

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_294646

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*
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
:ÿÿÿÿÿÿÿÿÿ^^@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿbb : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs

a
(__inference_dropout_layer_call_fn_298429

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_294938x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
D
(__inference_re_lu_4_layer_call_fn_298414

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_294766i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294538

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
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
 *o:t
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
:ÿÿÿÿÿÿÿÿÿk
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
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
b
F__inference_activation_layer_call_and_return_conditional_losses_294818

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
L
0__inference_max_pooling2d_3_layer_call_fn_298241

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_294371
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Ñ
6__inference_batch_normalization_5_layer_call_fn_298503

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_294585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
$__inference_signature_wrapper_297449
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

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	$

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_294134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
!
_user_specified_name	input_1
Ú 
´
#__inference__update_step_xla_296861
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ì

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294244

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä
D
(__inference_dropout_layer_call_fn_298424

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_294775i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
¡
)__inference_conv2d_4_layer_call_fn_298327

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_294745x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_294371

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
a
C__inference_dropout_layer_call_and_return_conditional_losses_298434

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
]
A__inference_re_lu_layer_call_and_return_conditional_losses_294634

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿbb :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
Ú 
´
#__inference__update_step_xla_296720
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_294447

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_298145

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

ÿ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_294679

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_298044

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
D
(__inference_re_lu_3_layer_call_fn_298313

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_294733i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
B
&__inference_re_lu_layer_call_fn_298010

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_294634h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿbb :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
³


D__inference_conv2d_4_layer_call_and_return_conditional_losses_294745

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È 
±
#__inference__update_step_xla_296344
gradient
variable: !
readvariableop_resource:	 #
readvariableop_1_resource: +
sub_2_readvariableop_resource: +
sub_3_readvariableop_resource: ¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
 *ÍÌÌ=N
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes
: 
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
 *o:N
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0X
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes
: 
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
 *¿Ö3Q
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
	
Ï
4__inference_batch_normalization_layer_call_fn_297956

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_294168
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä

&__inference_dense_layer_call_fn_298466

inputs
unknown:	$
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_294798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
ö
¡
)__inference_conv2d_3_layer_call_fn_298226

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_294712x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

)__inference_conv2d_1_layer_call_fn_298024

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_294646w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^^@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿbb : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_2_layer_call_fn_298158

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294320
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä"
Û
#__inference__update_step_xla_296814
gradient$
variable:!
readvariableop_resource:	 #
readvariableop_1_resource: 9
sub_2_readvariableop_resource:9
sub_3_readvariableop_resource:¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0g
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=\
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*(
_output_shapes
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0M
SquareSquaregradient*
T0*(
_output_shapes
:|
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*(
_output_shapes
:*
dtype0i
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:\
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*(
_output_shapes
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*(
_output_shapes
:*
dtype0f
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*(
_output_shapes
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*(
_output_shapes
:*
dtype0`
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3_
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*(
_output_shapes
:]
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*(
_output_shapes
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*1
_input_shapes 
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ë
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_294733

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³


D__inference_conv2d_3_layer_call_and_return_conditional_losses_298236

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_294275

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_4_layer_call_fn_298373

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_294503
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_294320

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú 
´
#__inference__update_step_xla_296955
gradient
variable:	!
readvariableop_resource:	 #
readvariableop_1_resource: ,
sub_2_readvariableop_resource:	,
sub_3_readvariableop_resource:	¢AssignAddVariableOp¢AssignAddVariableOp_1¢AssignSubVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢Sqrt_1/ReadVariableOp¢sub_2/ReadVariableOp¢sub_3/ReadVariableOp^
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
 *w¾?J
Pow_1PowCast_2/x:output:0Cast:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?F
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
 *  ?H
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
:*
dtype0Z
sub_2Subgradientsub_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=O
mul_1Mul	sub_2:z:0mul_1/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOpAssignAddVariableOpsub_2_readvariableop_resource	mul_1:z:0^sub_2/ReadVariableOp*
_output_shapes
 *
dtype0@
SquareSquaregradient*
T0*
_output_shapes	
:o
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes	
:*
dtype0\
sub_3Sub
Square:y:0sub_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:O
mul_2Mul	sub_3:z:0mul_2/y:output:0*
T0*
_output_shapes	
:
AssignAddVariableOp_1AssignAddVariableOpsub_3_readvariableop_resource	mul_2:z:0^sub_3/ReadVariableOp*
_output_shapes
 *
dtype0
ReadVariableOp_2ReadVariableOpsub_2_readvariableop_resource^AssignAddVariableOp*
_output_shapes	
:*
dtype0Y
mul_3MulReadVariableOp_2:value:0truediv:z:0*
T0*
_output_shapes	
:
Sqrt_1/ReadVariableOpReadVariableOpsub_3_readvariableop_resource^AssignAddVariableOp_1*
_output_shapes	
:*
dtype0S
Sqrt_1SqrtSqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3R
add_1AddV2
Sqrt_1:y:0add_1/y:output:0*
T0*
_output_shapes	
:P
	truediv_1RealDiv	mul_3:z:0	add_1:z:0*
T0*
_output_shapes	
:f
AssignSubVariableOpAssignSubVariableOpvariabletruediv_1:z:0*
_output_shapes
 *
dtype0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*$
_input_shapes
:: : : : : *
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
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
½
L
0__inference_max_pooling2d_1_layer_call_fn_298039

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_294219
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_298906CustomGradient-296136"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default¡
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿôô<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ò

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
ö
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
Ö
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
Ê
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
À
Ztrace_0
[trace_1
\trace_2
]trace_32Õ
$__inference_htc_layer_call_fn_294896
$__inference_htc_layer_call_fn_297526
$__inference_htc_layer_call_fn_297603
$__inference_htc_layer_call_fn_295357º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zZtrace_0z[trace_1z\trace_2z]trace_3
¬
^trace_0
_trace_1
`trace_2
atrace_32Á
?__inference_htc_layer_call_and_return_conditional_losses_297748
?__inference_htc_layer_call_and_return_conditional_losses_297914
?__inference_htc_layer_call_and_return_conditional_losses_295462
?__inference_htc_layer_call_and_return_conditional_losses_295567º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z^trace_0z_trace_1z`trace_2zatrace_3
ÌBÉ
!__inference__wrapped_model_294134input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

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
Ý
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
¥
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	3gamma
4beta
5moving_mean
6moving_variance"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

7kernel
8bias
!_jit_compiled_convolution_op"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	 axis
	9gamma
:beta
;moving_mean
<moving_variance"
_tf_keras_layer
«
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses

=kernel
>bias
!­_jit_compiled_convolution_op"
_tf_keras_layer
«
®	variables
¯trainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer
ñ
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
	ºaxis
	?gamma
@beta
Amoving_mean
Bmoving_variance"
_tf_keras_layer
«
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses

Ckernel
Dbias
!Ç_jit_compiled_convolution_op"
_tf_keras_layer
«
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
ñ
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses
	Ôaxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance"
_tf_keras_layer
«
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses

Ikernel
Jbias
!á_jit_compiled_convolution_op"
_tf_keras_layer
«
â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"
_tf_keras_layer
ñ
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
ì__call__
+í&call_and_return_all_conditional_losses
	îaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance"
_tf_keras_layer
«
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses
û_random_generator"
_tf_keras_layer
«
ü	variables
ýtrainable_variables
þregularization_losses
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Okernel
Pbias"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
trace_02Í
__inference_test_step_295760¬
£²
FullArgSpec'
args
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
í
trace_02Î
__inference_train_step_297370¬
£²
FullArgSpec'
args
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
-
serving_default"
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
.:,@2htc/conv2d_2/kernel
 :2htc/conv2d_2/bias
.:,2htc/batch_normalization_2/gamma
-:+2htc/batch_normalization_2/beta
6:4 (2%htc/batch_normalization_2/moving_mean
::8 (2)htc/batch_normalization_2/moving_variance
/:-2htc/conv2d_3/kernel
 :2htc/conv2d_3/bias
.:,2htc/batch_normalization_3/gamma
-:+2htc/batch_normalization_3/beta
6:4 (2%htc/batch_normalization_3/moving_mean
::8 (2)htc/batch_normalization_3/moving_variance
/:-2htc/conv2d_4/kernel
 :2htc/conv2d_4/bias
.:,2htc/batch_normalization_4/gamma
-:+2htc/batch_normalization_4/beta
6:4 (2%htc/batch_normalization_4/moving_mean
::8 (2)htc/batch_normalization_4/moving_variance
#:!	$2htc/dense/kernel
:2htc/dense/bias
-:+2htc/batch_normalization_5/gamma
,:*2htc/batch_normalization_5/beta
5:3 (2%htc/batch_normalization_5/moving_mean
9:7 (2)htc/batch_normalization_5/moving_variance
¶
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
Þ
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
ñBî
$__inference_htc_layer_call_fn_294896input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ëBè
$__inference_htc_layer_call_fn_297526x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ëBè
$__inference_htc_layer_call_fn_297603x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ñBî
$__inference_htc_layer_call_fn_295357input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
?__inference_htc_layer_call_and_return_conditional_losses_297748x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
?__inference_htc_layer_call_and_return_conditional_losses_297914x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
?__inference_htc_layer_call_and_return_conditional_losses_295462input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
?__inference_htc_layer_call_and_return_conditional_losses_295567input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Î
c0
1
2
3
4
5
6
7
8
 9
¡10
¢11
£12
¤13
¥14
¦15
§16
¨17
©18
ª19
«20
¬21
­22
®23
¯24
°25
±26
²27
³28
´29
µ30
¶31
·32
¸33
¹34
º35
»36
¼37
½38
¾39
¿40
À41
Á42
Â43
Ã44
Ä45
Å46
Æ47
Ç48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
î
0
1
2
3
 4
¢5
¤6
¦7
¨8
ª9
¬10
®11
°12
²13
´14
¶15
¸16
º17
¼18
¾19
À20
Â21
Ä22
Æ23"
trackable_list_wrapper
î
0
1
2
3
¡4
£5
¥6
§7
©8
«9
­10
¯11
±12
³13
µ14
·15
¹16
»17
½18
¿19
Á20
Ã21
Å22
Ç23"
trackable_list_wrapper
¿2¼¹
®²ª
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
í
Ítrace_02Î
'__inference_conv2d_layer_call_fn_297923¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÍtrace_0

Îtrace_02é
B__inference_conv2d_layer_call_and_return_conditional_losses_297933¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÎtrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ô
Ôtrace_02Õ
.__inference_max_pooling2d_layer_call_fn_297938¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÔtrace_0

Õtrace_02ð
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_297943¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÕtrace_0
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
¸
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ý
Ûtrace_0
Ütrace_12¢
4__inference_batch_normalization_layer_call_fn_297956
4__inference_batch_normalization_layer_call_fn_297969³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÛtrace_0zÜtrace_1

Ýtrace_0
Þtrace_12Ø
O__inference_batch_normalization_layer_call_and_return_conditional_losses_297987
O__inference_batch_normalization_layer_call_and_return_conditional_losses_298005³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÝtrace_0zÞtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ì
ätrace_02Í
&__inference_re_lu_layer_call_fn_298010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zätrace_0

åtrace_02è
A__inference_re_lu_layer_call_and_return_conditional_losses_298015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zåtrace_0
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
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
ëtrace_02Ð
)__inference_conv2d_1_layer_call_fn_298024¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zëtrace_0

ìtrace_02ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_298034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zìtrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ö
òtrace_02×
0__inference_max_pooling2d_1_layer_call_fn_298039¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zòtrace_0

ótrace_02ò
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_298044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zótrace_0
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
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
á
ùtrace_0
útrace_12¦
6__inference_batch_normalization_1_layer_call_fn_298057
6__inference_batch_normalization_1_layer_call_fn_298070³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zùtrace_0zútrace_1

ûtrace_0
ütrace_12Ü
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_298088
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_298106³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zûtrace_0zütrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_re_lu_1_layer_call_fn_298111¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_re_lu_1_layer_call_and_return_conditional_losses_298116¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_conv2d_2_layer_call_fn_298125¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ë
D__inference_conv2d_2_layer_call_and_return_conditional_losses_298135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
®	variables
¯trainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
ö
trace_02×
0__inference_max_pooling2d_2_layer_call_fn_298140¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ò
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_298145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
á
trace_0
trace_12¦
6__inference_batch_normalization_2_layer_call_fn_298158
6__inference_batch_normalization_2_layer_call_fn_298171³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ü
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298189
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298207³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
»	variables
¼trainable_variables
½regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
î
 trace_02Ï
(__inference_re_lu_2_layer_call_fn_298212¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0

¡trace_02ê
C__inference_re_lu_2_layer_call_and_return_conditional_losses_298217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¡trace_0
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
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
ï
§trace_02Ð
)__inference_conv2d_3_layer_call_fn_298226¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0

¨trace_02ë
D__inference_conv2d_3_layer_call_and_return_conditional_losses_298236¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¨trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
ö
®trace_02×
0__inference_max_pooling2d_3_layer_call_fn_298241¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0

¯trace_02ò
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_298246¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¯trace_0
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
¸
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
á
µtrace_0
¶trace_12¦
6__inference_batch_normalization_3_layer_call_fn_298259
6__inference_batch_normalization_3_layer_call_fn_298272³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0z¶trace_1

·trace_0
¸trace_12Ü
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298290
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298308³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0z¸trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
î
¾trace_02Ï
(__inference_re_lu_3_layer_call_fn_298313¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0

¿trace_02ê
C__inference_re_lu_3_layer_call_and_return_conditional_losses_298318¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¿trace_0
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
¸
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
ï
Åtrace_02Ð
)__inference_conv2d_4_layer_call_fn_298327¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0

Ætrace_02ë
D__inference_conv2d_4_layer_call_and_return_conditional_losses_298337¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÆtrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
â	variables
ãtrainable_variables
äregularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
ö
Ìtrace_02×
0__inference_max_pooling2d_4_layer_call_fn_298342¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0

Ítrace_02ò
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_298347¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÍtrace_0
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
¸
Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
è	variables
étrainable_variables
êregularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
á
Ótrace_0
Ôtrace_12¦
6__inference_batch_normalization_4_layer_call_fn_298360
6__inference_batch_normalization_4_layer_call_fn_298373³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÓtrace_0zÔtrace_1

Õtrace_0
Ötrace_12Ü
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298391
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298409³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÕtrace_0zÖtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
î
Ütrace_02Ï
(__inference_re_lu_4_layer_call_fn_298414¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÜtrace_0

Ýtrace_02ê
C__inference_re_lu_4_layer_call_and_return_conditional_losses_298419¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÝtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
Å
ãtrace_0
ätrace_12
(__inference_dropout_layer_call_fn_298424
(__inference_dropout_layer_call_fn_298429³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zãtrace_0zätrace_1
û
åtrace_0
ætrace_12À
C__inference_dropout_layer_call_and_return_conditional_losses_298434
C__inference_dropout_layer_call_and_return_conditional_losses_298446³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zåtrace_0zætrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
ü	variables
ýtrainable_variables
þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
î
ìtrace_02Ï
(__inference_flatten_layer_call_fn_298451¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zìtrace_0

ítrace_02ê
C__inference_flatten_layer_call_and_return_conditional_losses_298457¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zítrace_0
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
¸
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ì
ótrace_02Í
&__inference_dense_layer_call_fn_298466¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zótrace_0

ôtrace_02è
A__inference_dense_layer_call_and_return_conditional_losses_298477¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zôtrace_0
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
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
á
útrace_0
ûtrace_12¦
6__inference_batch_normalization_5_layer_call_fn_298490
6__inference_batch_normalization_5_layer_call_fn_298503³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zútrace_0zûtrace_1

ütrace_0
ýtrace_12Ü
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298523
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298557³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zütrace_0zýtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_activation_layer_call_fn_298562¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_activation_layer_call_and_return_conditional_losses_298567¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
âBß
__inference_test_step_295760imageslabels"¬
£²
FullArgSpec'
args
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ãBà
__inference_train_step_297370imageslabels"¬
£²
FullArgSpec'
args
jself
jimages
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_297449input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
3:1@2Adam/m/htc/conv2d_2/kernel
3:1@2Adam/v/htc/conv2d_2/kernel
%:#2Adam/m/htc/conv2d_2/bias
%:#2Adam/v/htc/conv2d_2/bias
3:12&Adam/m/htc/batch_normalization_2/gamma
3:12&Adam/v/htc/batch_normalization_2/gamma
2:02%Adam/m/htc/batch_normalization_2/beta
2:02%Adam/v/htc/batch_normalization_2/beta
4:22Adam/m/htc/conv2d_3/kernel
4:22Adam/v/htc/conv2d_3/kernel
%:#2Adam/m/htc/conv2d_3/bias
%:#2Adam/v/htc/conv2d_3/bias
3:12&Adam/m/htc/batch_normalization_3/gamma
3:12&Adam/v/htc/batch_normalization_3/gamma
2:02%Adam/m/htc/batch_normalization_3/beta
2:02%Adam/v/htc/batch_normalization_3/beta
4:22Adam/m/htc/conv2d_4/kernel
4:22Adam/v/htc/conv2d_4/kernel
%:#2Adam/m/htc/conv2d_4/bias
%:#2Adam/v/htc/conv2d_4/bias
3:12&Adam/m/htc/batch_normalization_4/gamma
3:12&Adam/v/htc/batch_normalization_4/gamma
2:02%Adam/m/htc/batch_normalization_4/beta
2:02%Adam/v/htc/batch_normalization_4/beta
(:&	$2Adam/m/htc/dense/kernel
(:&	$2Adam/v/htc/dense/kernel
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
ÛBØ
'__inference_conv2d_layer_call_fn_297923inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_conv2d_layer_call_and_return_conditional_losses_297933inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
âBß
.__inference_max_pooling2d_layer_call_fn_297938inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_297943inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ùBö
4__inference_batch_normalization_layer_call_fn_297956inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
4__inference_batch_normalization_layer_call_fn_297969inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_batch_normalization_layer_call_and_return_conditional_losses_297987inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_batch_normalization_layer_call_and_return_conditional_losses_298005inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÚB×
&__inference_re_lu_layer_call_fn_298010inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_re_lu_layer_call_and_return_conditional_losses_298015inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_conv2d_1_layer_call_fn_298024inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_298034inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
äBá
0__inference_max_pooling2d_1_layer_call_fn_298039inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_298044inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ûBø
6__inference_batch_normalization_1_layer_call_fn_298057inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_1_layer_call_fn_298070inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_298088inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_298106inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÜBÙ
(__inference_re_lu_1_layer_call_fn_298111inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_1_layer_call_and_return_conditional_losses_298116inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_conv2d_2_layer_call_fn_298125inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_298135inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
äBá
0__inference_max_pooling2d_2_layer_call_fn_298140inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_298145inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ûBø
6__inference_batch_normalization_2_layer_call_fn_298158inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_2_layer_call_fn_298171inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298189inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298207inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÜBÙ
(__inference_re_lu_2_layer_call_fn_298212inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_2_layer_call_and_return_conditional_losses_298217inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_conv2d_3_layer_call_fn_298226inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_298236inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
äBá
0__inference_max_pooling2d_3_layer_call_fn_298241inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_298246inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ûBø
6__inference_batch_normalization_3_layer_call_fn_298259inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_3_layer_call_fn_298272inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298290inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298308inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÜBÙ
(__inference_re_lu_3_layer_call_fn_298313inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_3_layer_call_and_return_conditional_losses_298318inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_conv2d_4_layer_call_fn_298327inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_298337inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
äBá
0__inference_max_pooling2d_4_layer_call_fn_298342inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_298347inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ûBø
6__inference_batch_normalization_4_layer_call_fn_298360inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_4_layer_call_fn_298373inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298391inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298409inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÜBÙ
(__inference_re_lu_4_layer_call_fn_298414inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_4_layer_call_and_return_conditional_losses_298419inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
íBê
(__inference_dropout_layer_call_fn_298424inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
íBê
(__inference_dropout_layer_call_fn_298429inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_298434inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_298446inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÜBÙ
(__inference_flatten_layer_call_fn_298451inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_flatten_layer_call_and_return_conditional_losses_298457inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÚB×
&__inference_dense_layer_call_fn_298466inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_dense_layer_call_and_return_conditional_losses_298477inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ûBø
6__inference_batch_normalization_5_layer_call_fn_298490inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_5_layer_call_fn_298503inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298523inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298557inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_activation_layer_call_fn_298562inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_activation_layer_call_and_return_conditional_losses_298567inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ½
!__inference__wrapped_model_294134$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQ:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿôô
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ©
F__inference_activation_layer_call_and_return_conditional_losses_298567_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
+__inference_activation_layer_call_fn_298562T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿó
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2980889:;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "F¢C
<9
tensor_0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2981069:;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "F¢C
<9
tensor_0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Í
6__inference_batch_normalization_1_layer_call_fn_2980579:;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª ";8
unknown+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Í
6__inference_batch_normalization_1_layer_call_fn_2980709:;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª ";8
unknown+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@õ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298189?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "G¢D
=:
tensor_0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_298207?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "G¢D
=:
tensor_0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
6__inference_batch_normalization_2_layer_call_fn_298158?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "<9
unknown,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÏ
6__inference_batch_normalization_2_layer_call_fn_298171?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "<9
unknown,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿõ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298290EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "G¢D
=:
tensor_0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_298308EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "G¢D
=:
tensor_0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
6__inference_batch_normalization_3_layer_call_fn_298259EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "<9
unknown,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÏ
6__inference_batch_normalization_3_layer_call_fn_298272EFGHN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "<9
unknown,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿõ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298391KLMNN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "G¢D
=:
tensor_0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_298409KLMNN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "G¢D
=:
tensor_0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
6__inference_batch_normalization_4_layer_call_fn_298360KLMNN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "<9
unknown,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÏ
6__inference_batch_normalization_4_layer_call_fn_298373KLMNN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "<9
unknown,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298523iSTRQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¾
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_298557iSTRQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_5_layer_call_fn_298490^STRQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_5_layer_call_fn_298503^STRQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!
unknownÿÿÿÿÿÿÿÿÿñ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2979873456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "F¢C
<9
tensor_0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ñ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2980053456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "F¢C
<9
tensor_0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
4__inference_batch_normalization_layer_call_fn_2979563456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª ";8
unknown+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
4__inference_batch_normalization_layer_call_fn_2979693456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª ";8
unknown+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ »
D__inference_conv2d_1_layer_call_and_return_conditional_losses_298034s787¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿbb 
ª "4¢1
*'
tensor_0ÿÿÿÿÿÿÿÿÿ^^@
 
)__inference_conv2d_1_layer_call_fn_298024h787¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿbb 
ª ")&
unknownÿÿÿÿÿÿÿÿÿ^^@¼
D__inference_conv2d_2_layer_call_and_return_conditional_losses_298135t=>7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_2_layer_call_fn_298125i=>7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "*'
unknownÿÿÿÿÿÿÿÿÿ½
D__inference_conv2d_3_layer_call_and_return_conditional_losses_298236uCD8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_3_layer_call_fn_298226jCD8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "*'
unknownÿÿÿÿÿÿÿÿÿ½
D__inference_conv2d_4_layer_call_and_return_conditional_losses_298337uIJ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_4_layer_call_fn_298327jIJ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "*'
unknownÿÿÿÿÿÿÿÿÿ½
B__inference_conv2d_layer_call_and_return_conditional_losses_297933w129¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôô
ª "6¢3
,)
tensor_0ÿÿÿÿÿÿÿÿÿíí 
 
'__inference_conv2d_layer_call_fn_297923l129¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôô
ª "+(
unknownÿÿÿÿÿÿÿÿÿíí ©
A__inference_dense_layer_call_and_return_conditional_losses_298477dOP0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ$
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
&__inference_dense_layer_call_fn_298466YOP0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ$
ª "!
unknownÿÿÿÿÿÿÿÿÿ¼
C__inference_dropout_layer_call_and_return_conditional_losses_298434u<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¼
C__inference_dropout_layer_call_and_return_conditional_losses_298446u<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dropout_layer_call_fn_298424j<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*'
unknownÿÿÿÿÿÿÿÿÿ
(__inference_dropout_layer_call_fn_298429j<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*'
unknownÿÿÿÿÿÿÿÿÿ°
C__inference_flatten_layer_call_and_return_conditional_losses_298457i8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ$
 
(__inference_flatten_layer_call_fn_298451^8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ""
unknownÿÿÿÿÿÿÿÿÿ$ä
?__inference_htc_layer_call_and_return_conditional_losses_295462 $123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ¢G
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿôô
ª

trainingp ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ä
?__inference_htc_layer_call_and_return_conditional_losses_295567 $123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ¢G
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿôô
ª

trainingp",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Þ
?__inference_htc_layer_call_and_return_conditional_losses_297748$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD¢A
*¢'
%"
xÿÿÿÿÿÿÿÿÿôô
ª

trainingp ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Þ
?__inference_htc_layer_call_and_return_conditional_losses_297914$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD¢A
*¢'
%"
xÿÿÿÿÿÿÿÿÿôô
ª

trainingp",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¾
$__inference_htc_layer_call_fn_294896$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ¢G
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿôô
ª

trainingp "!
unknownÿÿÿÿÿÿÿÿÿ¾
$__inference_htc_layer_call_fn_295357$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQJ¢G
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿôô
ª

trainingp"!
unknownÿÿÿÿÿÿÿÿÿ¸
$__inference_htc_layer_call_fn_297526$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD¢A
*¢'
%"
xÿÿÿÿÿÿÿÿÿôô
ª

trainingp "!
unknownÿÿÿÿÿÿÿÿÿ¸
$__inference_htc_layer_call_fn_297603$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQD¢A
*¢'
%"
xÿÿÿÿÿÿÿÿÿôô
ª

trainingp"!
unknownÿÿÿÿÿÿÿÿÿé
#__inference_internal_grad_fn_298906ÁÈ¢Ä
¼¢¸

 
'$
result_grads_0 

result_grads_1 

result_grads_2 

result_grads_3 
'$
result_grads_4 @

result_grads_5@

result_grads_6@

result_grads_7@
(%
result_grads_8@

result_grads_9

result_grads_10

result_grads_11
*'
result_grads_12

result_grads_13

result_grads_14

result_grads_15
*'
result_grads_16

result_grads_17

result_grads_18

result_grads_19
!
result_grads_20	$

result_grads_21

result_grads_22

result_grads_23
(%
result_grads_24 

result_grads_25 

result_grads_26 

result_grads_27 
(%
result_grads_28 @

result_grads_29@

result_grads_30@

result_grads_31@
)&
result_grads_32@

result_grads_33

result_grads_34

result_grads_35
*'
result_grads_36

result_grads_37

result_grads_38

result_grads_39
*'
result_grads_40

result_grads_41

result_grads_42

result_grads_43
!
result_grads_44	$

result_grads_45

result_grads_46

result_grads_47
ª "óï
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
"
	tensor_24 

	tensor_25 

	tensor_26 

	tensor_27 
"
	tensor_28 @

	tensor_29@

	tensor_30@

	tensor_31@
# 
	tensor_32@

	tensor_33

	tensor_34

	tensor_35
$!
	tensor_36

	tensor_37

	tensor_38

	tensor_39
$!
	tensor_40

	tensor_41

	tensor_42

	tensor_43

	tensor_44	$

	tensor_45

	tensor_46

	tensor_47õ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_298044¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
0__inference_max_pooling2d_1_layer_call_fn_298039R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿõ
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_298145¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
0__inference_max_pooling2d_2_layer_call_fn_298140R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿõ
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_298246¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
0__inference_max_pooling2d_3_layer_call_fn_298241R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿõ
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_298347¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
0__inference_max_pooling2d_4_layer_call_fn_298342R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_297943¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Í
.__inference_max_pooling2d_layer_call_fn_297938R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¶
C__inference_re_lu_1_layer_call_and_return_conditional_losses_298116o7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "4¢1
*'
tensor_0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_re_lu_1_layer_call_fn_298111d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ")&
unknownÿÿÿÿÿÿÿÿÿ@¸
C__inference_re_lu_2_layer_call_and_return_conditional_losses_298217q8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_re_lu_2_layer_call_fn_298212f8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "*'
unknownÿÿÿÿÿÿÿÿÿ¸
C__inference_re_lu_3_layer_call_and_return_conditional_losses_298318q8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_re_lu_3_layer_call_fn_298313f8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "*'
unknownÿÿÿÿÿÿÿÿÿ¸
C__inference_re_lu_4_layer_call_and_return_conditional_losses_298419q8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "5¢2
+(
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_re_lu_4_layer_call_fn_298414f8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "*'
unknownÿÿÿÿÿÿÿÿÿ´
A__inference_re_lu_layer_call_and_return_conditional_losses_298015o7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿbb 
ª "4¢1
*'
tensor_0ÿÿÿÿÿÿÿÿÿbb 
 
&__inference_re_lu_layer_call_fn_298010d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿbb 
ª ")&
unknownÿÿÿÿÿÿÿÿÿbb Ë
$__inference_signature_wrapper_297449¢$123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿôô"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
__inference_test_step_295760u(123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQ-./0E¢B
;¢8
!
images ôô

labels 
ª "
 ú
__inference_train_step_297370Ø123456789:;<=>?@ABCDEFGHIJKLMNOPSTRQcd ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇ)*+,E¢B
;¢8
!
images ôô

labels 
ª "
 