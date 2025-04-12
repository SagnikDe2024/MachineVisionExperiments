

The amou

Let's assume that the there are 4 CNN layers and kernel size is 5.
Then assuming that each layer is of type
Conv -> BatchNorm -> Activation -> MaxPool(2,2)

Assuming down sampling by 2.

The kernel coverage of 4th CNN layer on itself is 5. 
The kernel coverage of 3rd CNN layer is (5 * 2 - 1)+5 = 14.
The kernel coverage of 2nd CNN layer is (14 * 2 - 1)+5 = 32.
The kernel coverage of 1st CNN layer is (32 * 2 - 1) + 5 = 68.



The kernel coverage from 1st layer is 5.
The kernel coverage from 2nd layer is 5 * 2 = 10. 
The kernel coverage from 3rd layer is 5 * 2^2 = 20.
The kernel coverage from 1st layer is 5 * 2^3 = 40.

(5*2-1+5)*2-1

Let's consider a CNN based encoder. The encoder downsamples by the ratio $r$ every layer
 and there are $n$ layers.
So for kernel size $k$ we have coverage 
K for it self.
$ C(0) = k $ and $ C(m+1) = C(m) \times r + 2 \times k - 2 $.
Solving we get   

$$ C(m) = \frac{(2 - 2 \times k +  r^m (k \times r + 2 \times k - 2) )}{r-1} $$






