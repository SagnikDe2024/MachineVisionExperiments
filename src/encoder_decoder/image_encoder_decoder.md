# Image encoder and decoder

## Basic Idea

### Kernel coverage

Let's assume that the CNN layers are like the one below:

Conv -> BatchNorm -> Activation -> MaxPool

Let this layer down samples by the ratio $r$  and there are $n$ layers.
So for kernel size $k$ we have coverage $k$ for itself. Counting from back to front for the encoder we have for the kernel $k$ to have coverage
$k$. Let that be CNN layer 0. 
Thus :

$ C(0) = k $ 

For any layer $m$ we end up with the recurrence relation $ C(m+1) = C(m) \times r + 2 \times k - 2 $.


Solving it we get   

$$ C(m) = \frac{(2 - 2 \times k +  r^m (k \times r + k - 2) )}{r-1} $$

replacing $r$ with 2 we get 

$$ C(m) = \frac{(2 - 2 \times k +  2^m (k \times 2 + k - 2) )}{2-1} $$

Let coverage be $ 2^n $ and $ m = l -1 $ then we 

$$ 2^n = 2 - 2 \times k +   2^{l-1} (k \times 2 + k - 2) $$

solving for $k$ 

$$ k = \frac{2 (2^l + 2^n - 2)}{3 \times 2^l - 4}$$

or 

$$ k = \frac{2 (2^l + s - 2)}{3 \times 2^l - 4}$$

For the simple case of $m = 3$ where it is down sampled 4 times (down sampling happens after CNN layer). Assuming down sampling by $r = 2$ 
every layer we get the coverage for kernel size $k = 3$ as 52.  
The total down sampling $d$ is related to $r$ by     
$$ r^{m+1} = d \implies r = d^{\frac{1}{m+1}} $$  
Then one can eliminate $r$ and from the coverage formula becomes   

$$ C(m) = \frac{(2 - 2 \times k +  d^{m/(m+1)} (k \times d^{1/(m+1)} + k - 2) )}{d^{1/(m+1)}-1} $$  

Now we put $C(m) = 128 $ 






Since the image is sliced into blocks the other information being fed into the 
system is of type

1) Height and Width of the image.
2) The cropping location top, left.
3) A hash of the image.

