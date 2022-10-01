# install
# > install.packages("swirl")
# > library("swirl")
# > swirl()
# 
# 
#  -- Typing skip() allows you to skip the current question.
# | -- Typing play() lets you experiment with R on your own; swirl will ignore what you do...
# | -- UNTIL you type nxt() which will regain swirl's attention.
# | -- Typing bye() causes swirl to exit. Your progress will be saved.
# | -- Typing main() returns you to swirl's main menu.
# | -- Typing info() displays these options again.

# assign to x 
# x <- ???

# c function - conctenate or combine
# create vector
# c()
# you can concantenate to further vectors
# assigned vectors can have arithmetic principles worked upon

#?function()
# r's built in help files
# function's functions

# ^ powers
# sqrt()
# abs()
# 
# | When given two vectors of the same length, R simply performs the specified arithmetic operation (`+`, `-`, `*`, etc.)
# | element-by-element. If the vectors are of different lengths, R 'recycles' the shorter vector until it is the same
# | length as the longer vector.
# 
# ...
# 
# |==================================================================================                             |  74%
# | When we did z * 2 + 100 in our earlier example, z was a vector of length 3, but technically 2 and 100 are each vectors
# | of length 1.
# 
# ...
# 
# |=====================================================================================                          |  76%
# | Behind the scenes, R is 'recycling' the 2 to make a vector of 2s and the 100 to make a vector of 100s. In other words,
# | when you ask R to compute z * 2 + 100, what it really computes is this: z * c(2, 2, 2) + c(100, 100, 100)
# 
# To see another example of how this vector 'recycling' works, try adding c(1, 2, 3, 4) and c(0, 10). Don't worry about
# | saving the result in a new variable.
# 
# > 
# > c(1,2,3,4) + c(0, 10)
# [1]  1 12  3 14
# 
# | Type c(1, 2, 3, 4) + c(0, 10, 100) to see how R handles adding two vectors, when the shorter vector's length does not
# | divide evenly into the longer vector's length. Don't worry about assigning the result to a variable.
# 
# > c(1,2,3,4) + c(0,10,100)
# [1]   1  12 103   4
# Warning message:
#   In c(1, 2, 3, 4) + c(0, 10, 100) :
#   longer object length is not a multiple of shorter object length
# 
# | In many programming environments, the up arrow will cycle through previous commands. Try hitting the up arrow on your
# | keyboard until you get to this command (z * 2 + 100), then change 100 to 1000 and hit Enter. If the up arrow doesn't
# | work for you, just type the corrected 

# getwd() - current working directory
# ls(), list all objects in local workspace

#List all the files in your working directory using list.files() or dir().
# list.files {base}

# args() - Argument List of a Function
# > args(list.files)
# function (path = ".", pattern = NULL, all.files = FALSE, full.names = FALSE, 
#           recursive = FALSE, ignore.case = FALSE, include.dirs = FALSE, 
#           no.. = FALSE) 

# dir.create() to create a directory in the current working directory 

# | Set your working directory to "testdir" with the setwd() command.
#| Create a file in your working directory called "mytest.R" using the file.create() function.
# Check to see if "mytest.R" exists in the working directory using the file.exists() function.
#| Access information about the file "mytest.R" by using file.info().

# > file.info("mytest.R")
# size isdir mode               mtime               ctime               atime exe
# mytest.R    0 FALSE  666 2022-08-24 21:38:39 2022-08-24 21:38:39 2022-08-24 21:38:39  no

# |=======================================================================                                        |  64%
# | You can use the $ operator --- e.g., file.info("mytest.R")$mode --- to grab specific items.
# 
# ...
# 
# |==========================================================================                                     |  67%
# | Change the name of the file "mytest.R" to "mytest2.R" by using file.rename().
# file.remove('mytest.R')
# | Make a copy of "mytest2.R" called "mytest3.R" using file.copy().
# | Provide the relative path to the file "mytest3.R" by using file.path().

#
#| You can use file.path to construct file and directory paths that are independent of the operating system your R code
# | is running on. Pass 'folder1' and 'folder2' as arguments to file.path to make a platform-independent pathname.
# 
# > file.path("folder1", "folder2")
# [1] "folder1/folder2"

# |===========================================================================================                    |  82%
# | Take a look at the documentation for dir.create by entering ?dir.create . Notice the 'recursive' argument. In order to
# | create nested directories, 'recursive' must be set to TRUE.
#

# | The simplest way to create a sequence of numbers in R is by using the `:` operator. Type 1:20 to see how it works.
# 
# | That gave us every integer between (and including) 1 and 20. We could also use it to create a sequence of real
# | numbers. For example, try pi:10.
# go backwards - 10:1
# 
# > pi:10
# [1] 3.141593 4.141593 5.141593 6.141593 7.141593 8.141593 9.141593
# ?`:`in the case of an operator like the colon used above, you must enclose the symbol in backticks
# seq() does exactly the same thing as the `:` operator
# seq(0, 10, by=0.5)
# > seq(5, 10, len=30)
# [1]  5.000000  5.172414  5.344828  5.517241  5.689655  5.862069  6.034483  6.206897  6.379310  6.551724  6.724138
# [12]  6.896552  7.068966  7.241379  7.413793  7.586207  7.758621  7.931034  8.103448  8.275862  8.448276  8.620690
# [23]  8.793103  8.965517  9.137931  9.310345  9.482759  9.655172  9.827586 10.000000
# length()
# 
# |===============================================================                                                |  57%
# | Let's pretend we don't know the length of my_seq, but we want to generate a sequence of integers from 1 to N, where N
# | represents the length of the my_seq vector. In other words, we want a new vector (1, 2, 3, ...) that is the same
# | length as my_seq.
# 
# ...
# 
# |====================================================================                                           |  61%
# | There are several ways we could do this. One possibility is to combine the `:` operator and the length() function like
# | this: 1:length(my_seq). Give that a try.
# 
# > 1:length(my_seq)
# [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

#   |========================================================================                                       |  65%
# | Another option is to use seq(along.with = my_seq). Give that a try.
# 
# > seq(along.with= my_seq)
# [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# seq_along()

#  rep(0, times=40)
#  [1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

# |=====================================================================================================          |  91%
# | If instead we want our vector to contain 10 repetitions of the vector (0, 1, 2), we can do rep(c(0, 1, 2), times =
#                                                                                                    | 10). Go ahead.
# 
# > rep(c(0,1,2), times=10)
# [1] 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2
# 
# | Keep up the great work!
#   
#   |==========================================================================================================     |  96%
# | Finally, let's say that rather than repeating the vector (0, 1, 2) over and over again, we want our vector to contain
# | 10 zeros, then 10 ones, then 10 twos. We can do this with the `each` argument. Try rep(c(0, 1, 2), each = 10).
# 
# > rep(c(0,1,2), each=10)
#  [1] 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2


# | The simplest and most common data structure in R is the vector.
# | Vectors come in two different flavors: atomic vectors and lists. An atomic vector contains exactly one data type,
# | whereas a list may contain multiple data types.
# | In previous lessons, we dealt entirely with numeric vectors, which are one type of atomic vector. Other types of
# | atomic vectors include logical, character, integer, and complex. In this lesson, we'll take a closer look at logical
# | and character vectors.
# | Logical vectors can contain the values TRUE, FALSE, and NA (for 'not available'). These values are generated as the
# | result of logical 'conditions'. Let's experiment with some simple conditions.
# 
# 
# | Use tf <- num_vect < 1 to assign the result of num_vect < 1 to a variable called tf.
# 
# > tf <- num_vect < 1
# 
# > tf
# [1]  TRUE FALSE  TRUE FALSE
# | The `<` and `>=` symbols in these examples are called 'logical operators'. Other logical operators include `>`, `<=`,
# | `==` for exact equality, and `!=` for inequality.
# | If we have two logical expressions, A and B, we can ask whether at least one is TRUE with A | B (logical 'or' a.k.a.
#                                                                                                    | 'union') or whether they are both TRUE with A & B (logical 'and' a.k.a. 'intersection'). Lastly, !A is the negation of
# | A and is TRUE when A is FALSE and vice versa.
# 
# |===================================================================                                            |  61%
# | Right now, my_char is a character vector of length 3. Let's say we want to join the elements of my_char together into
# | one continuous character string (i.e. a character vector of length 1). We can do this using the paste() function.
# 
# ...
# 
#   |======================================================================                                         |  63%
# | Type paste(my_char, collapse = " ") now. Make sure there's a space between the double quotes in the `collapse`
# | argument. You'll see why in a second.
# 
# | In the simplest case, we can join two character vectors that are each of length 1 (i.e. join two words). Try
# | paste("Hello", "world!", sep = " "), where the `sep` argument tells R that we want to separate the joined elements
# | with a single space.]
# 
# | Use paste(1:3, c("X", "Y", "Z"), sep = "") to see what happens when we join two vectors of equal length using paste().
# 
# > paste(1:3, c("X", "Y", "Z"), sep="")
# [1] "1X" "2Y" "3Z"
# 
# LETTERS is a predefined variable in R containing a
# | character vector of all 26 letters in the English alphabet.
# | Missing values play an important role in statistics and data analysis. Often, missing values must not be ignored, but
# | rather they should be carefully studied to see if there's an underlying pattern or cause for their missingness.
# 
# ...
# 
#   |======                                                                                                         |   5%
# | In R, NA is used to represent any value that is 'not available' or 'missing' (in the statistical sense). In this
# | lesson, we'll explore missing values further.
# 
# ...
# 
# |===========                                                                                                    |  10%
# | Any operation involving NA generally yields NA as the result. To illustrate, let's create a vector c(44, NA, 5, NA)
# | and assign it to a variable x.
# NA values don't multiply

# | To make things a little more interesting, lets create a vector containing 1000 draws from a standard normal
# | distribution with y <- rnorm(1000).
# 
# > y <- rnorm(1000)
# 
# is.na()

# | In our previous discussion of logical operators, we introduced the `==` operator as a method of testing for equality
# | between two objects. So, you might think the expression my_data == NA yields the same results as is.na(). Give it a
# | try.
# 
# > my_data == na
# Error: object 'na' not found
# > my_data == NA
# [1] NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA
# [39] NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA
# [77] NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA NA
# 
# | Nice work!
#   
#   |===================================================================                                            |  60%
# | The reason you got a vector of all NAs is that NA is not really a value, but just a placeholder for a quantity that is
# | not available. Therefore the logical expression is incomplete and R has no choice but to return a vector of the same
# | length as my_data that contains all NAs.

# NaN - not anumber
# Inf

# x[1:10] subletting vectors
# Index vectors come in - pos int, neg int, logical, character strings

# x[is.na(x)] = vector of all NA in x
# x[!is.na(x)] = vector of all none-NA in x


# |=====================================                                                                          |  33%
# | Type y[y > 0] to see that we get all of the positive elements of y, which are also the positive elements of our
# | original vector x.
# 
# > y[y > 0]
# [1] 0.6907889 0.2790635 0.9362678 0.2479663 1.2833904 2.9238459 0.2783149 0.3017936 1.1376034 1.2804736 1.7690890
# [12] 0.1462693
# 
# | You are really on a roll!
#   
#   |========================================                                                                       |  36%
# | You might wonder why we didn't just start with x[x > 0] to isolate the positive elements of x. Try that now to see
# | why.
# 
# > x[x > 0]
#  [1]        NA 0.6907889        NA        NA 0.2790635        NA        NA        NA        NA        NA        NA
# [12]        NA        NA 0.9362678 0.2479663        NA        NA 1.2833904        NA 2.9238459        NA 0.2783149
# [23]        NA        NA 0.3017936 1.1376034        NA 1.2804736 1.7690890        NA        NA 0.1462693
# 
# | Great job!
# 
#   |===========================================                                                                    |  38%
# | Since NA is not a value, but rather a placeholder for an unknown quantity, the expression NA > 0 evaluates to NA.
# | Hence we get a bunch of NAs mixed in with our positive numbers when we do this.

# get specific multiple subset - x[c(a, b, c)]
# |============================================================                                                   |  54%
# | It's important that when using integer vectors to subset our vector x, we stick with the set of indexes {1, 2, ...,
# | 40} since x only has 40 elements. What happens if we ask for the zeroth element of x (i.e. x[0])? Give it a try.
# 
# > x[0]
# numeric(0)
# 
# | All that practice is paying off!
# 
#   |===============================================================                                                |  56%
# | As you might expect, we get nothing useful. Unfortunately, R doesn't prevent us from doing this. What if we ask for
# | the 3000th element of x? Try it out.
# 
# > x[3000]
# [1] NA
# 
# | All that practice is paying off!

# omit from vector - x[c(-a, -b)] or x[-c(a, b)]

# # named elements
# > vect <- c(foo = 11, bar =2, norf=na)
# > vect
# foo  bar norf 
# 11    2   NA 
# > names(vect)
# [1] "foo"  "bar"  "norf"
# | Alternatively, we can create an unnamed vector vect2 with c(11, 2, NA). Do that now.
# 
# > 
#   > vect2 <- c(11,2, NA)
# 
# | Your dedication is inspiring!
#   
#   |===========================================================================================                    |  82%
# | Then, we can add the `names` attribute to vect2 after the fact with names(vect2) <- c("foo", "bar", "norf"). Go ahead.
# 
# > names(vect2) <- c("foo", "bar", "norf")
# 
# 
# > identical(vect, vect2)
# [1] TRUE
# 
# |====================================================================================================           |  90%
# | Now, back to the matter of subsetting a vector by named elements. Which of the following commands do you think would
# | give us the second element of vect?
#   
#   1: vect["2"]
# 2: vect["bar"]
# 3: vect[bar]
# 
# Selection: 2
# 
# | You are really on a roll!
#   
#   |======================================================================================================         |  92%
# | Now, try it out.
# 
# > 
#   > vect["bar"]
# bar 
# 2 
# 
# | That's the answer I was looking for.
# 
#   |=========================================================================================================      |  95%
# | Likewise, we can specify a vector of names with vect[c("foo", "bar")]. Try it out.
# 
# > vect[c("foo", "bar")]
# foo bar 
#  11   2 
#  
#  > my_vector <- 1:20
# > my_vector
#  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# | The dim() function tells us the 'dimensions' of an object. What happens if we do dim(my_vector)? Give it a try.
# 
# > dim(my_vector)
# NULL
# 
# | Ah! That's what we wanted. But, what happens if we give my_vector a `dim` attribute? Let's give it a try. Type
# | dim(my_vector) <- c(4, 5).
# 
# > dim(my_vector) <- c(4,5)
# 
# 
# | Use dim(my_vector) to confirm that we've set the `dim` attribute correctly.
# 
# > dim(my_vector)
# [1] 4 5
# 
# 
# > my_vector
# [,1] [,2] [,3] [,4] [,5]
# [1,]    1    5    9   13   17
# [2,]    2    6   10   14   18
# [3,]    3    7   11   15   19
# [4,]    4    8   12   16   20
# 
# 
# > patients <- c("Bill", "Gina", "Kelly", "Sean")
# > cbind(patients, my_matrix)
# patients                       
# [1,] "Bill"   "1" "5" "9"  "13" "17"
# [2,] "Gina"   "2" "6" "10" "14" "18"
# [3,] "Kelly"  "3" "7" "11" "15" "19"
# [4,] "Sean"   "4" "8" "12" "16" "20"
# 
# |====================================================================                                           |  61%
# | Something is fishy about our result! It appears that combining the character vector with our matrix of numbers caused
# | everything to be enclosed in double quotes. This means we're left with a matrix of character strings, which is no
# | good.
# 
# ...
# 
#   |=======================================================================                                        |  64%
# | If you remember back to the beginning of this lesson, I told you that matrices can only contain ONE class of data.
# | Therefore, when we tried to combine a character vector with a numeric matrix, R was forced to 'coerce' the numbers to
# | characters, hence the double quotes.
# 
#   |==========================================================================                                     |  67%
# | This is called 'implicit coercion', because we didn't ask for it. It just happened. But why didn't R just convert the
# | names of our patients to numbers? I'll let you ponder that question on your own.
# 
# | Now, let's confirm it's actually a matrix by using the class() function. Type class(my_vector) to see what I mean.
# 
# > class(my_vector)
# 
# |=============================================================================                                  |  69%
# | So, we're still left with the question of how to include the names of our patients in the table without destroying the
# | integrity of our numeric data. Try the following -- my_data <- data.frame(patients, my_matrix)
# 
# > my_data <- data.frame(patients, my_matrix)
# 
# | That's the answer I was looking for.
# 
# |================================================================================                               |  72%
# | Now view the contents of my_data to see what we've come up with.
# 
# > my_data
#   patients X1 X2 X3 X4 X5
# 1     Bill  1  5  9 13 17
# 2     Gina  2  6 10 14 18
# 3    Kelly  3  7 11 15 19
# 4     Sean  4  8 12 16 20
# 
# | You are quite good my friend!
#   |===================================================================================                            |  75%
# | It looks like the data.frame() function allowed us to store our character vector of names right alongside our matrix
# | of numbers. That's exactly what we were hoping for!
#   
#   ...
# 
# |======================================================================================                         |  78%
# | Behind the scenes, the data.frame() function takes any number of arguments and returns a single object of class
# | `data.frame` that is composed of the original objects.
# 
# ...
# 
# |=========================================================================================                      |  81%
# | Let's confirm this by calling the class() function on our newly created data frame.
# 
# > class(my_data)
# [1] "data.frame"
# 
# | Great job!
# 
#   |============================================================================================                   |  83%
# | It's also possible to assign names to the individual rows and columns of a data frame, which presents another possible
# | way of determining which row of values in our table belongs to each patient.
# 
# ...
# 
# |================================================================================================               |  86%
# | However, since we've already solved that problem, let's solve a different problem by assigning names to the columns of
# | our data frame so that we know what type of measurement each column represents.
# 
# ...
# 
# |===================================================================================================            |  89%
# | Since we have six columns (including patient names), we'll need to first create a vector containing one element for
# | each column. Create a character vector called cnames that contains the following values (in order) -- "patient",
# | "age", "weight", "bp", "rating", "test".
# 
# > 
# > cnames <- c("patient", "age", "weight", "bp", "rating", "test")
# 
# | You are doing so well!
# 
#   |======================================================================================================         |  92%
# | Now, use the colnames() function to set the `colnames` attribute for our data frame. This is similar to the way we
# | used the dim() function earlier in this lesson.
# 
# > colnames(my_data) <- cnames
# 
# | Keep up the great work!
# 
#   |=========================================================================================================      |  94%
# | Let's see if that got the job done. Print the contents of my_data.
# 
# > my_data
# patient age weight bp rating test
# 1    Bill   1      5  9     13   17
# 2    Gina   2      6 10     14   18
# 3   Kelly   3      7 11     15   19
# 4    Sean   4      8 12     16   20
# 
# | You are doing so well!
#   
#   
#   | You can use the `&` operator to evaluate AND across a vector. The `&&` version of AND only evaluates the first member
# | of a vector. Let's test both for practice. Type the expression TRUE & c(TRUE, FALSE, FALSE).
# 
# > TRUE && c(TRUE, FALSE, FALSE)
# [1] TRUE
# Warning message:
# In TRUE && c(TRUE, FALSE, FALSE) :
#   'length(x) = 3 > 1' in coercion to 'logical(1)'
#   
#   | The OR operator follows a similar set of rules. The `|` version of OR evaluates OR across an entire vector, while the
# | `||` version of OR only evaluates the first member of a vector.
# 
# | Now let's try out the non-vectorized version of the OR operator. Type the expression TRUE || c(TRUE, FALSE, FALSE).
# 
# > TRUE || c(TRUE, FALSE, FALSE)
# [1] TRUE
# 
# | You are really on a roll!
#   
#   |============================================================                                                   |  54%
# | Logical operators can be chained together just like arithmetic operators. The expressions: `6 != 10 && FALSE && 1 >=
#   | 2` or `TRUE || 5 < 9.3 || FALSE` are perfectly normal to see.
# 
# The function isTRUE() takes one argument. If that argument evaluates to TRUE, the function will return TRUE.
# 
# identical()
# 
# The xor() function stands for exclusive OR.
# | If one argument evaluates to TRUE and one argument evaluates to FALSE, then this function will return TRUE, otherwise
# | it will return FALSE.
# 
# | For the next few questions, we're going to need to create a vector of integers called ints. Create this vector by
# | typing: ints <- sample(10)
# 
# > ints <- sample(10)
# 
# > ints
#  [1]  5  1  3  8 10  2  7  9  6  4
# 
# | You got it!
# 
#   |============================================================================================                   |  83%
# | The vector `ints` is a random sampling of integers from 1 to 10 without replacement. Let's say we wanted to ask some
# | logical questions about contents of ints. If we type ints > 5, we will get a logical vector corresponding to whether
# | each element of ints is greater than 5. Try typing: ints > 5
# 
# > ints > 5
# [1] FALSE FALSE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE
# 
# The which() function takes a logical vector
# | as an argument and returns the indices of the vector that are TRUE. For example which(c(TRUE, FALSE, TRUE)) would
# | return the vector c(1, 3).
# 
# ...
# 
# |================================================================================================               |  87%
# | Use the which() function to find the indices of ints that are greater than 7.
# 
# > which(ints > 7)
# [1] 4 5 8
# 
# any()
# all()
# 
# > Sys.Date()
# [1] "2022-08-25"
# mean()
# 
# function without () shows it's source code
# 
# 
# |=============================================                                                                  |  41%
# | You can also explicitly specify arguments in a function. When you explicitly designate argument values by name, the
# | ordering of the arguments becomes unimportant. You can try this out by typing: remainder(divisor = 11, num = 5).
# 
# > remainder(divisor = 11, num = 5)
# [1] 5
# 
# | You got it right!
#   
#   |================================================                                                               |  43%
# | As you can see, there is a significant difference between remainder(11, 5) and remainder(divisor = 11, num = 5)!
#   
#   ...
# 
# |==================================================                                                             |  45%
# | R can also partially match arguments. Try typing remainder(4, div = 2) to see this feature in action.
# 
# > remainder(4, div = 2)
# [1] 0
# 
# sd - standard dev
# 
# 
# | Let's use the evaluate function to explore how anonymous functions work. For the first argument of the evaluate
# | function we're going to write a tiny function that fits on one line. In the second argument we'll pass some data to
# | the tiny anonymous function in the first argument.
# 
# ...
# 
# |======================================================================                                         |  63%
# | Type the following command and then we'll discuss how it works: evaluate(function(x){x+1}, 6)
# 
# > evaluate(function(x){x+1}, 6)
# [1] 7
# 
# 
# 
# paste {base}	R Documentation
# Concatenate Strings
# Description
# Concatenate vectors after converting to character.
# 
# 
# | You're familiar with adding, subtracting, multiplying, and dividing numbers in R. To do this you use the +, -, *, and
# | / symbols. These symbols are called binary operators because they take two inputs, an input from the left and an input
# | from the right.
# 
# ...
# 
#   |======================================================================================================         |  92%
# | In R you can define your own binary operators. In the next script I'll show you how.
# 
# ...
# 
# # "%p%" <- function(left, right){ # Remember to add arguments!
# paste(left, right)
# }

| In this lesson, you'll learn how to use lapply() and sapply(), the two most important members of R's *apply family of
| functions, also known as loop functions.

...

|==                                                                                                             |   2%
| These powerful functions, along with their close relatives (vapply() and tapply(), among others) offer a concise and
| convenient means of implementing the Split-Apply-Combine strategy for data analysis.

...

|====                                                                                                           |   4%
| Each of the *apply functions will SPLIT up some data into smaller pieces, APPLY a function to each piece, then COMBINE
| the results. A more detailed discussion of this strategy is found in Hadley Wickham's Journal of Statistical Software
| paper titled 'The Split-Apply-Combine Strategy for Data Analysis'.

...

  |=======                                                                                                        |   6%
| Throughout this lesson, we'll use the Flags dataset from the UCI Machine Learning Repository. This dataset contains
| details of various nations and their flags. More information may be found here:
  | http://archive.ics.uci.edu/ml/datasets/Flags

...

|=========                                                                                                      |   8%
| Let's jump right in so you can get a feel for how these special functions work!

...

  |===========                                                                                                    |  10%
| I've stored the dataset in a variable called flags. Type head(flags) to preview the first six lines (i.e. the 'head')
| of the dataset.

> head(flags)
name landmass zone area population language religion bars stripes colours red green blue gold white black
1    Afghanistan        5    1  648         16       10        2    0       3       5   1     1    0    1     1     1
2        Albania        3    1   29          3        6        6    0       0       3   1     0    0    1     0     1
3        Algeria        4    1 2388         20        8        2    2       0       3   1     1    0    0     1     0
4 American-Samoa        6    3    0          0        1        1    0       0       5   1     0    1    1     1     0
5        Andorra        3    1    0          0        6        0    3       0       3   1     0    1    1     0     0
6         Angola        4    2 1247          7       10        5    0       2       3   1     0    0    1     0     1
orange mainhue circles crosses saltires quarters sunstars crescent triangle icon animate text topleft botright
1      0   green       0       0        0        0        1        0        0    1       0    0   black    green
2      0     red       0       0        0        0        1        0        0    0       1    0     red      red
3      0   green       0       0        0        0        1        1        0    0       0    0   green    white
4      1    blue       0       0        0        0        0        0        1    1       1    0    blue      red
5      0    gold       0       0        0        0        0        0        0    0       0    0    blue      red
6      0     red       0       0        0        0        1        0        0    1       0    0     red    black

| Perseverance, that's the answer.

  |=============                                                                                                  |  12%
| You may need to scroll up to see all of the output. Now, let's check out the dimensions of the dataset using
| dim(flags).

> dim(flags)
[1] 194  30

| You are really on a roll!
  
  |================                                                                                               |  14%
| This tells us that there are 194 rows, or observations, and 30 columns, or variables. Each observation is a country
| and each variable describes some characteristic of that country or its flag. To open a more complete description of
| the dataset in a separate text file, type viewinfo() when you are back at the prompt (>).

...

|==================                                                                                             |  16%
| As with any dataset, we'd like to know in what format the variables have been stored. In other words, what is the
| 'class' of each variable? What happens if we do class(flags)? Try it out.

> class(flags)
[1] "data.frame"

| Great job!

  |====================                                                                                           |  18%
| That just tells us that the entire dataset is stored as a 'data.frame', which doesn't answer our question. What we
| really need is to call the class() function on each individual column. While we could do this manually (i.e. one
                                                                                                          | column at a time) it's much faster if we can automate the process. Sounds like a loop!

...

  |======================                                                                                         |  20%
| The lapply() function takes a list as input, applies a function to each element of the list, then returns a list of
| the same length as the original one. Since a data frame is really just a list of vectors (you can see this with
| as.list(flags)), we can use lapply() to apply the class() function to each column of the flags dataset. Let's see it
| in action!
  
  ...

|========================                                                                                       |  22%
| Type cls_list <- lapply(flags, class) to apply the class() function to each column of the flags dataset and store the
| result in a variable called cls_list. Note that you just supply the name of the function you want to apply (i.e.
                                                                                                              | class), without the usual parentheses after it.

> cls_list <-lapply(flags, class)

| You are amazing!
  
  |===========================                                                                                    |  24%
| Type cls_list to view the result.

> cls_list
$name
[1] "character"

$landmass
[1] "integer"

$zone
[1] "integer"

$area
[1] "integer"

$population
[1] "integer"

$language
[1] "integer"

$religion
[1] "integer"

$bars
[1] "integer"

$stripes
[1] "integer"

$colours
[1] "integer"

$red
[1] "integer"

$green
[1] "integer"

$blue
[1] "integer"

$gold
[1] "integer"

$white
[1] "integer"

$black
[1] "integer"

$orange
[1] "integer"

$mainhue
[1] "character"

$circles
[1] "integer"

$crosses
[1] "integer"

$saltires
[1] "integer"

$quarters
[1] "integer"

$sunstars
[1] "integer"

$crescent
[1] "integer"

$triangle
[1] "integer"

$icon
[1] "integer"

$animate
[1] "integer"

$text
[1] "integer"

$topleft
[1] "character"

$botright
[1] "character"


| You are amazing!
  
  |=============================                                                                                  |  26%
| The 'l' in 'lapply' stands for 'list'. Type class(cls_list) to confirm that lapply() returned a list.

> class(cls_list)
[1] "list"

| Your dedication is inspiring!
  
  |===============================                                                                                |  28%
| As expected, we got a list of length 30 -- one element for each variable/column. The output would be considerably more
| compact if we could represent it as a vector instead of a list.

...

|=================================                                                                              |  30%
| You may remember from a previous lesson that lists are most helpful for storing multiple classes of data. In this
| case, since every element of the list returned by lapply() is a character vector of length one (i.e. "integer" and
                                                                                                  | "vector"), cls_list can be simplified to a character vector. To do this manually, type as.character(cls_list).

> as.character(cls_list)
[1] "character" "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"  
[10] "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "character"
[19] "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"  
[28] "integer"   "character" "character"

| Perseverance, that's the answer.

  |====================================                                                                           |  32%
| sapply() allows you to automate this process by calling lapply() behind the scenes, but then attempting to simplify
| (hence the 's' in 'sapply') the result for you. Use sapply() the same way you used lapply() to get the class of each
| column of the flags dataset and store the result in cls_vect. If you need help, type ?sapply to bring up the
| documentation.

> ?sapply
> sapply(flags, class)
       name    landmass        zone        area  population    language    religion        bars     stripes     colours 
"character"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" 
        red       green        blue        gold       white       black      orange     mainhue     circles     crosses 
  "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" "character"   "integer"   "integer" 
   saltires    quarters    sunstars    crescent    triangle        icon     animate        text     topleft    botright 
  "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" "character" "character" 

| You almost had it, but not quite. Try again. Or, type info() for more options.

| Type cls_vect <- sapply(flags, class) to store the column classes in a character vector called cls_vect.

> cls_vect <- sapply(flags, class)

| That's the answer I was looking for.

|======================================                                                                         |  34%
| Use class(cls_vect) to confirm that sapply() simplified the result to a character vector.

> class(cls_vect)
[1] "character"

| Excellent work!
  
  |========================================                                                                       |  36%
| In general, if the result is a list where every element is of length one, then sapply() returns a vector. If the
| result is a list where every element is a vector of the same length (> 1), sapply() returns a matrix. If sapply()
| can't figure things out, then it just returns a list, no different from what lapply() would give you.

...

  |==========================================                                                                     |  38%
| Let's practice using lapply() and sapply() some more!
  
  ...

|============================================                                                                   |  40%
| Columns 11 through 17 of our dataset are indicator variables, each representing a different color. The value of the
| indicator variable is 1 if the color is present in a country's flag and 0 otherwise.

...

  |===============================================                                                                |  42%
| Therefore, if we want to know the total number of countries (in our dataset) with, for example, the color orange on
| their flag, we can just add up all of the 1s and 0s in the 'orange' column. Try sum(flags$orange) to see this.

> sum(flags$orange)
[1] 26

| Excellent work!

  |=================================================                                                              |  44%
| Now we want to repeat this operation for each of the colors recorded in the dataset.

...

  |===================================================                                                            |  46%
| First, use flag_colors <- flags[, 11:17] to extract the columns containing the color data and store them in a new data
| frame called flag_colors. (Note the comma before 11:17. This subsetting command tells R that we want all rows, but
| only columns 11 through 17.)

> flag_colors <- flags[, 11:17]

| Your dedication is inspiring!

  |=====================================================                                                          |  48%
| Use the head() function to look at the first 6 lines of flag_colors.

> head(flag_colors)
  red green blue gold white black orange
1   1     1    0    1     1     1      0
2   1     0    0    1     0     1      0
3   1     1    0    0     1     0      0
4   1     0    1    1     1     0      1
5   1     0    1    1     0     0      0
6   1     0    0    1     0     1      0

| Excellent job!

  |========================================================                                                       |  50%
| To get a list containing the sum of each column of flag_colors, call the lapply() function with two arguments. The
| first argument is the object over which we are looping (i.e. flag_colors) and the second argument is the name of the
| function we wish to apply to each column (i.e. sum). Remember that the second argument is just the name of the
| function with no parentheses, etc.

> lapply(flag_colors, sum)
$red
[1] 153

$green
[1] 91

$blue
[1] 99

$gold
[1] 91

$white
[1] 146

$black
[1] 52

$orange
[1] 26


| You are really on a roll!

  |==========================================================                                                     |  52%
| This tells us that of the 194 flags in our dataset, 153 contain the color red, 91 contain green, 99 contain blue, and
| so on.

...

  |============================================================                                                   |  54%
| The result is a list, since lapply() always returns a list. Each element of this list is of length one, so the result
| can be simplified to a vector by calling sapply() instead of lapply(). Try it now.

> sapply(flag_colors, sum)
   red  green   blue   gold  white  black orange 
   153     91     99     91    146     52     26 

| You got it right!

  |==============================================================                                                 |  56%
| Perhaps it's more informative to find the proportion of flags (out of 194) containing each color. Since each column is
| just a bunch of 1s and 0s, the arithmetic mean of each column will give us the proportion of 1s. (If it's not clear
| why, think of a simpler situation where you have three 1s and two 0s -- (1 + 1 + 1 + 0 + 0)/5 = 3/5 = 0.6).

...

  |================================================================                                               |  58%
| Use sapply() to apply the mean() function to each column of flag_colors. Remember that the second argument to sapply()
| should just specify the name of the function (i.e. mean) that you want to apply.

> sapply(flag_colors, mean)
      red     green      blue      gold     white     black    orange 
0.7886598 0.4690722 0.5103093 0.4690722 0.7525773 0.2680412 0.1340206 

| You're the best!
                                                                                                      
                                                                                                      |===================================================================                                            |  60%
                                                                                                    | In the examples we've looked at so far, sapply() has been able to simplify the result to vector. That's because each
                                                                                                    | element of the list returned by lapply() was a vector of length one. Recall that sapply() instead returns a matrix
                                                                                                    | when each element of the list returned by lapply() is a vector of the same length (> 1).
                                                                                                    
                                                                                                    ...
                                                                                                    
                                                                                                    |=====================================================================                                          |  62%
                                                                                                    | To illustrate this, let's extract columns 19 through 23 from the flags dataset and store the result in a new data
| frame called flag_shapes. flag_shapes <- flags[, 19:23] will do it.

> flag_shapes <- flags[, 19:23]

| Nice work!

  |=======================================================================                                        |  64%
| Each of these columns (i.e. variables) represents the number of times a particular shape or design appears on a
| country's flag. We are interested in the minimum and maximum number of times each shape or design appears.
                                                                                                    
                                                                                                    ...
                                                                                                    
                                                                                                    |=========================================================================                                      |  66%
                                                                                                    | The range() function returns the minimum and maximum of its first argument, which should be a numeric vector. Use
                                                                                                    | lapply() to apply the range function to each column of flag_shapes. Don't worry about storing the result in a new
| variable. By now, we know that lapply() always returns a list.

> lapply(flag_shapes, mean)
$circles
[1] 0.1701031

$crosses
[1] 0.1494845

$saltires
[1] 0.09278351

$quarters
[1] 0.1494845

$sunstars
[1] 1.386598


| Try again. Getting it right on the first try is boring anyway! Or, type info() for more options.

| Try lapply(flag_shapes, range) to apply the range() function to each column of flag_shapes.

> lapply(flag_shapes, range)
$circles
[1] 0 4

$crosses
[1] 0 2

$saltires
[1] 0 1

$quarters
[1] 0 4

$sunstars
[1]  0 50


| You're the best!
                                                                                                      
                                                                                                      |===========================================================================                                    |  68%
                                                                                                    | Do the same operation, but using sapply() and store the result in a variable called shape_mat.
                                                                                                    
                                                                                                    > shape_mat <- sapply(flag_shapes, range)
                                                                                                    
                                                                                                    | You are really on a roll!
                                                                                                      
                                                                                                      |==============================================================================                                 |  70%
                                                                                                    | View the contents of shape_mat.
                                                                                                    
                                                                                                    > shape_mat
                                                                                                    circles crosses saltires quarters sunstars
                                                                                                    [1,]       0       0        0        0        0
                                                                                                    [2,]       4       2        1        4       50
                                                                                                    
                                                                                                    | That's a job well done!

  |================================================================================                               |  72%
| Each column of shape_mat gives the minimum (row 1) and maximum (row 2) number of times its respective shape appears in
| different flags.

...

  |==================================================================================                             |  74%
| Use the class() function to confirm that shape_mat is a matrix.

> class(shape_mat)
[1] "matrix" "array" 

| All that hard work is paying off!

  |====================================================================================                           |  76%
| As we've seen, sapply() always attempts to simplify the result given by lapply(). It has been successful in doing so
                                                                                                    | for each of the examples we've looked at so far. Let's look at an example where sapply() can't figure out how to
| simplify the result and thus returns a list, no different from lapply().

...

  |=======================================================================================                        |  78%
| When given a vector, the unique() function returns a vector with all duplicate elements removed. In other words,
| unique() returns a vector of only the 'unique' elements. To see how it works, try unique(c(3, 4, 5, 5, 5, 6, 6)).

> unique(c(3,4,5,5,5,6,6))
[1] 3 4 5 6

| You nailed it! Good job!

  |=========================================================================================                      |  80%
| We want to know the unique values for each variable in the flags dataset. To accomplish this, use lapply() to apply
| the unique() function to each column in the flags dataset, storing the result in a variable called unique_vals.

> unique_vals <- lapply(unique(flags))
Error in match.fun(FUN) : argument "FUN" is missing, with no default
> unique_vals <- lapply(flags, unique)

| You got it right!

  |===========================================================================================                    |  82%
| Print the value of unique_vals to the console.

> unique_vals
$name
  [1] "Afghanistan"              "Albania"                  "Algeria"                  "American-Samoa"          
  [5] "Andorra"                  "Angola"                   "Anguilla"                 "Antigua-Barbuda"         
  [9] "Argentina"                "Argentine"                "Australia"                "Austria"                 
 [13] "Bahamas"                  "Bahrain"                  "Bangladesh"               "Barbados"                
 [17] "Belgium"                  "Belize"                   "Benin"                    "Bermuda"                 
 [21] "Bhutan"                   "Bolivia"                  "Botswana"                 "Brazil"                  
 [25] "British-Virgin-Isles"     "Brunei"                   "Bulgaria"                 "Burkina"                 
 [29] "Burma"                    "Burundi"                  "Cameroon"                 "Canada"                  
 [33] "Cape-Verde-Islands"       "Cayman-Islands"           "Central-African-Republic" "Chad"                    
 [37] "Chile"                    "China"                    "Colombia"                 "Comorro-Islands"         
 [41] "Congo"                    "Cook-Islands"             "Costa-Rica"               "Cuba"                    
 [45] "Cyprus"                   "Czechoslovakia"           "Denmark"                  "Djibouti"                
 [49] "Dominica"                 "Dominican-Republic"       "Ecuador"                  "Egypt"                   
 [53] "El-Salvador"              "Equatorial-Guinea"        "Ethiopia"                 "Faeroes"                 
 [57] "Falklands-Malvinas"       "Fiji"                     "Finland"                  "France"                  
 [61] "French-Guiana"            "French-Polynesia"         "Gabon"                    "Gambia"                  
 [65] "Germany-DDR"              "Germany-FRG"              "Ghana"                    "Gibraltar"               
 [69] "Greece"                   "Greenland"                "Grenada"                  "Guam"                    
 [73] "Guatemala"                "Guinea"                   "Guinea-Bissau"            "Guyana"                  
 [77] "Haiti"                    "Honduras"                 "Hong-Kong"                "Hungary"                 
 [81] "Iceland"                  "India"                    "Indonesia"                "Iran"                    
 [85] "Iraq"                     "Ireland"                  "Israel"                   "Italy"                   
 [89] "Ivory-Coast"              "Jamaica"                  "Japan"                    "Jordan"                  
 [93] "Kampuchea"                "Kenya"                    "Kiribati"                 "Kuwait"                  
 [97] "Laos"                     "Lebanon"                  "Lesotho"                  "Liberia"                 
[101] "Libya"                    "Liechtenstein"            "Luxembourg"               "Malagasy"                
[105] "Malawi"                   "Malaysia"                 "Maldive-Islands"          "Mali"                    
[109] "Malta"                    "Marianas"                 "Mauritania"               "Mauritius"               
[113] "Mexico"                   "Micronesia"               "Monaco"                   "Mongolia"                
[117] "Montserrat"               "Morocco"                  "Mozambique"               "Nauru"                   
[121] "Nepal"                    "Netherlands"              "Netherlands-Antilles"     "New-Zealand"             
[125] "Nicaragua"                "Niger"                    "Nigeria"                  "Niue"                    
[129] "North-Korea"              "North-Yemen"              "Norway"                   "Oman"                    
[133] "Pakistan"                 "Panama"                   "Papua-New-Guinea"         "Parguay"                 
[137] "Peru"                     "Philippines"              "Poland"                   "Portugal"                
[141] "Puerto-Rico"              "Qatar"                    "Romania"                  "Rwanda"                  
[145] "San-Marino"               "Sao-Tome"                 "Saudi-Arabia"             "Senegal"                 
[149] "Seychelles"               "Sierra-Leone"             "Singapore"                "Soloman-Islands"         
[153] "Somalia"                  "South-Africa"             "South-Korea"              "South-Yemen"             
[157] "Spain"                    "Sri-Lanka"                "St-Helena"                "St-Kitts-Nevis"          
[161] "St-Lucia"                 "St-Vincent"               "Sudan"                    "Surinam"                 
[165] "Swaziland"                "Sweden"                   "Switzerland"              "Syria"                   
[169] "Taiwan"                   "Tanzania"                 "Thailand"                 "Togo"                    
[173] "Tonga"                    "Trinidad-Tobago"          "Tunisia"                  "Turkey"                  
[177] "Turks-Cocos-Islands"      "Tuvalu"                   "UAE"                      "Uganda"                  
[181] "UK"                       "Uruguay"                  "US-Virgin-Isles"          "USA"                     
[185] "USSR"                     "Vanuatu"                  "Vatican-City"             "Venezuela"               
[189] "Vietnam"                  "Western-Samoa"            "Yugoslavia"               "Zaire"                   
[193] "Zambia"                   "Zimbabwe"                

$landmass
[1] 5 3 4 6 1 2

$zone
[1] 1 3 2 4

$area
  [1]   648    29  2388     0  1247  2777  7690    84    19     1   143    31    23   113    47  1099   600  8512     6
 [20]   111   274   678    28   474  9976     4   623  1284   757  9561  1139     2   342    51   115     9   128    43
 [39]    22    49   284  1001    21  1222    12    18   337   547    91   268    10   108   249   239   132  2176   109
 [58]   246    36   215   112    93   103  3268  1904  1648   435    70   301   323    11   372    98   181   583   236
 [77]    30  1760     3   587   118   333  1240  1031  1973  1566   447   783   140    41  1267   925   121   195   324
 [96]   212   804    76   463   407  1285   300   313    92   237    26  2150   196    72   637  1221    99   288   505
[115]    66  2506    63    17   450   185   945   514    57     5   164   781   245   178  9363 22402    15   912   256
[134]   905   753   391

$population
 [1]   16    3   20    0    7   28   15    8   90   10    1    6  119    9   35    4   24    2   11 1008    5   47   31
[24]   54   17   61   14  684  157   39   57  118   13   77   12   56   18   84   48   36   22   29   38   49   45  231
[47]  274   60

$language
 [1] 10  6  8  1  2  4  3  5  7  9

$religion
[1] 2 6 1 0 5 3 4 7

$bars
[1] 0 2 3 1 5

$stripes
 [1]  3  0  2  1  5  9 11 14  4  6 13  7

$colours
[1] 5 3 2 8 6 4 7 1

$red
[1] 1 0

$green
[1] 1 0

$blue
[1] 0 1

$gold
[1] 1 0

$white
[1] 1 0

$black
[1] 1 0

$orange
[1] 0 1

$mainhue
[1] "green"  "red"    "blue"   "gold"   "white"  "orange" "black"  "brown" 

$circles
[1] 0 1 4 2

$crosses
[1] 0 1 2

$saltires
[1] 0 1

$quarters
[1] 0 1 4

$sunstars
 [1]  1  0  6 22 14  3  4  5 15 10  7  2  9 50

$crescent
[1] 0 1

$triangle
[1] 0 1

$icon
[1] 1 0

$animate
[1] 0 1

$text
[1] 0 1

$topleft
[1] "black"  "red"    "green"  "blue"   "white"  "orange" "gold"  

$botright
[1] "green"  "red"    "white"  "black"  "blue"   "gold"   "orange" "brown" 


| Excellent work!

  |=============================================================================================                  |  84%
| Since unique_vals is a list, you can use what you've learned to determine the length of each element of unique_vals
                                                                                                    | (i.e. the number of unique values for each variable). Simplify the result, if possible. Hint: Apply the length()
                                                                                                    | function to each element of unique_vals.
                                                                                                    
                                                                                                    > length(unique_vals)
                                                                                                    [1] 30
                                                                                                    
                                                                                                    | That's not the answer I was looking for, but try again. Or, type info() for more options.

| Apply the length() function to each element of the unique_vals list using sapply(). Remember, no parentheses after the
| name of the function you are applying (i.e. length).

> sapply(unique_vals, length)
      name   landmass       zone       area population   language   religion       bars    stripes    colours        red 
       194          6          4        136         48         10          8          5         12          8          2 
     green       blue       gold      white      black     orange    mainhue    circles    crosses   saltires   quarters 
         2          2          2          2          2          2          8          4          3          2          3 
  sunstars   crescent   triangle       icon    animate       text    topleft   botright 
        14          2          2          2          2          2          7          8 

| Great job!

  |===============================================================================================                |  86%
| The fact that the elements of the unique_vals list are all vectors of *different* length poses a problem for sapply(),
| since there's no obvious way of simplifying the result.
                                                                                                    
                                                                                                    ...
                                                                                                    
                                                                                                    |==================================================================================================             |  88%
                                                                                                    | Use sapply() to apply the unique() function to each column of the flags dataset to see that you get the same
                                                                                                    | unsimplified list that you got from lapply().
                                                                                                    
                                                                                                    > sapply(flags, unique)
                                                                                                    $name
                                                                                                    [1] "Afghanistan"              "Albania"                  "Algeria"                  "American-Samoa"          
                                                                                                    [5] "Andorra"                  "Angola"                   "Anguilla"                 "Antigua-Barbuda"         
                                                                                                    [9] "Argentina"                "Argentine"                "Australia"                "Austria"                 
                                                                                                    [13] "Bahamas"                  "Bahrain"                  "Bangladesh"               "Barbados"                
                                                                                                    [17] "Belgium"                  "Belize"                   "Benin"                    "Bermuda"                 
                                                                                                    [21] "Bhutan"                   "Bolivia"                  "Botswana"                 "Brazil"                  
                                                                                                    [25] "British-Virgin-Isles"     "Brunei"                   "Bulgaria"                 "Burkina"                 
                                                                                                    [29] "Burma"                    "Burundi"                  "Cameroon"                 "Canada"                  
                                                                                                    [33] "Cape-Verde-Islands"       "Cayman-Islands"           "Central-African-Republic" "Chad"                    
                                                                                                    [37] "Chile"                    "China"                    "Colombia"                 "Comorro-Islands"         
                                                                                                    [41] "Congo"                    "Cook-Islands"             "Costa-Rica"               "Cuba"                    
                                                                                                    [45] "Cyprus"                   "Czechoslovakia"           "Denmark"                  "Djibouti"                
                                                                                                    [49] "Dominica"                 "Dominican-Republic"       "Ecuador"                  "Egypt"                   
                                                                                                    [53] "El-Salvador"              "Equatorial-Guinea"        "Ethiopia"                 "Faeroes"                 
                                                                                                    [57] "Falklands-Malvinas"       "Fiji"                     "Finland"                  "France"                  
                                                                                                    [61] "French-Guiana"            "French-Polynesia"         "Gabon"                    "Gambia"                  
                                                                                                    [65] "Germany-DDR"              "Germany-FRG"              "Ghana"                    "Gibraltar"               
                                                                                                    [69] "Greece"                   "Greenland"                "Grenada"                  "Guam"                    
                                                                                                    [73] "Guatemala"                "Guinea"                   "Guinea-Bissau"            "Guyana"                  
                                                                                                    [77] "Haiti"                    "Honduras"                 "Hong-Kong"                "Hungary"                 
                                                                                                    [81] "Iceland"                  "India"                    "Indonesia"                "Iran"                    
                                                                                                    [85] "Iraq"                     "Ireland"                  "Israel"                   "Italy"                   
                                                                                                    [89] "Ivory-Coast"              "Jamaica"                  "Japan"                    "Jordan"                  
                                                                                                    [93] "Kampuchea"                "Kenya"                    "Kiribati"                 "Kuwait"                  
                                                                                                    [97] "Laos"                     "Lebanon"                  "Lesotho"                  "Liberia"                 
                                                                                                    [101] "Libya"                    "Liechtenstein"            "Luxembourg"               "Malagasy"                
                                                                                                    [105] "Malawi"                   "Malaysia"                 "Maldive-Islands"          "Mali"                    
                                                                                                    [109] "Malta"                    "Marianas"                 "Mauritania"               "Mauritius"               
                                                                                                    [113] "Mexico"                   "Micronesia"               "Monaco"                   "Mongolia"                
                                                                                                    [117] "Montserrat"               "Morocco"                  "Mozambique"               "Nauru"                   
                                                                                                    [121] "Nepal"                    "Netherlands"              "Netherlands-Antilles"     "New-Zealand"             
                                                                                                    [125] "Nicaragua"                "Niger"                    "Nigeria"                  "Niue"                    
                                                                                                    [129] "North-Korea"              "North-Yemen"              "Norway"                   "Oman"                    
                                                                                                    [133] "Pakistan"                 "Panama"                   "Papua-New-Guinea"         "Parguay"                 
                                                                                                    [137] "Peru"                     "Philippines"              "Poland"                   "Portugal"                
                                                                                                    [141] "Puerto-Rico"              "Qatar"                    "Romania"                  "Rwanda"                  
                                                                                                    [145] "San-Marino"               "Sao-Tome"                 "Saudi-Arabia"             "Senegal"                 
                                                                                                    [149] "Seychelles"               "Sierra-Leone"             "Singapore"                "Soloman-Islands"         
                                                                                                    [153] "Somalia"                  "South-Africa"             "South-Korea"              "South-Yemen"             
                                                                                                    [157] "Spain"                    "Sri-Lanka"                "St-Helena"                "St-Kitts-Nevis"          
                                                                                                    [161] "St-Lucia"                 "St-Vincent"               "Sudan"                    "Surinam"                 
                                                                                                    [165] "Swaziland"                "Sweden"                   "Switzerland"              "Syria"                   
                                                                                                    [169] "Taiwan"                   "Tanzania"                 "Thailand"                 "Togo"                    
                                                                                                    [173] "Tonga"                    "Trinidad-Tobago"          "Tunisia"                  "Turkey"                  
                                                                                                    [177] "Turks-Cocos-Islands"      "Tuvalu"                   "UAE"                      "Uganda"                  
                                                                                                    [181] "UK"                       "Uruguay"                  "US-Virgin-Isles"          "USA"                     
                                                                                                    [185] "USSR"                     "Vanuatu"                  "Vatican-City"             "Venezuela"               
                                                                                                    [189] "Vietnam"                  "Western-Samoa"            "Yugoslavia"               "Zaire"                   
                                                                                                    [193] "Zambia"                   "Zimbabwe"                
                                                                                                    
                                                                                                    $landmass
                                                                                                    [1] 5 3 4 6 1 2
                                                                                                    
                                                                                                    $zone
                                                                                                    [1] 1 3 2 4
                                                                                                    
                                                                                                    $area
                                                                                                    [1]   648    29  2388     0  1247  2777  7690    84    19     1   143    31    23   113    47  1099   600  8512     6
                                                                                                    [20]   111   274   678    28   474  9976     4   623  1284   757  9561  1139     2   342    51   115     9   128    43
                                                                                                    [39]    22    49   284  1001    21  1222    12    18   337   547    91   268    10   108   249   239   132  2176   109
                                                                                                    [58]   246    36   215   112    93   103  3268  1904  1648   435    70   301   323    11   372    98   181   583   236
                                                                                                    [77]    30  1760     3   587   118   333  1240  1031  1973  1566   447   783   140    41  1267   925   121   195   324
                                                                                                    [96]   212   804    76   463   407  1285   300   313    92   237    26  2150   196    72   637  1221    99   288   505
                                                                                                    [115]    66  2506    63    17   450   185   945   514    57     5   164   781   245   178  9363 22402    15   912   256
                                                                                                    [134]   905   753   391
                                                                                                    
                                                                                                    $population
                                                                                                    [1]   16    3   20    0    7   28   15    8   90   10    1    6  119    9   35    4   24    2   11 1008    5   47   31
                                                                                                    [24]   54   17   61   14  684  157   39   57  118   13   77   12   56   18   84   48   36   22   29   38   49   45  231
                                                                                                    [47]  274   60
                                                                                                    
                                                                                                    $language
                                                                                                    [1] 10  6  8  1  2  4  3  5  7  9
                                                                                                    
                                                                                                    $religion
                                                                                                    [1] 2 6 1 0 5 3 4 7
                                                                                                    
                                                                                                    $bars
                                                                                                    [1] 0 2 3 1 5
                                                                                                    
                                                                                                    $stripes
                                                                                                    [1]  3  0  2  1  5  9 11 14  4  6 13  7
                                                                                                    
                                                                                                    $colours
                                                                                                    [1] 5 3 2 8 6 4 7 1
                                                                                                    
                                                                                                    $red
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $green
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $blue
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $gold
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $white
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $black
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $orange
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $mainhue
                                                                                                    [1] "green"  "red"    "blue"   "gold"   "white"  "orange" "black"  "brown" 
                                                                                                    
                                                                                                    $circles
                                                                                                    [1] 0 1 4 2
                                                                                                    
                                                                                                    $crosses
                                                                                                    [1] 0 1 2
                                                                                                    
                                                                                                    $saltires
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $quarters
                                                                                                    [1] 0 1 4
                                                                                                    
                                                                                                    $sunstars
                                                                                                    [1]  1  0  6 22 14  3  4  5 15 10  7  2  9 50
                                                                                                    
                                                                                                    $crescent
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $triangle
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $icon
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $animate
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $text
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $topleft
                                                                                                    [1] "black"  "red"    "green"  "blue"   "white"  "orange" "gold"  
                                                                                                    
                                                                                                    $botright
                                                                                                    [1] "green"  "red"    "white"  "black"  "blue"   "gold"   "orange" "brown" 
                                                                                                    
                                                                                                    
                                                                                                    | That's the answer I was looking for.

  |====================================================================================================           |  90%
| Occasionally, you may need to apply a function that is not yet defined, thus requiring you to write your own. Writing
| functions in R is beyond the scope of this lesson, but let's look at a quick example of how you might do so in the
                                                                                                    | context of loop functions.
                                                                                                    
                                                                                                    ...
                                                                                                    
                                                                                                    |======================================================================================================         |  92%
                                                                                                    | Pretend you are interested in only the second item from each element of the unique_vals list that you just created.
                                                                                                    | Since each element of the unique_vals list is a vector and we're not aware of any built-in function in R that returns
| the second element of a vector, we will construct our own function.

...

  |========================================================================================================       |  94%
| lapply(unique_vals, function(elem) elem[2]) will return a list containing the second item from each element of the
| unique_vals list. Note that our function takes one argument, elem, which is just a 'dummy variable' that takes on the
| value of each element of unique_vals, in turn.

> lapply(unique_vals, function(elem) elem[2])
$name
[1] "Albania"

$landmass
[1] 3

$zone
[1] 3

$area
[1] 29

$population
[1] 3

$language
[1] 6

$religion
[1] 6

$bars
[1] 2

$stripes
[1] 0

$colours
[1] 3

$red
[1] 0

$green
[1] 0

$blue
[1] 1

$gold
[1] 0

$white
[1] 0

$black
[1] 0

$orange
[1] 1

$mainhue
[1] "red"

$circles
[1] 1

$crosses
[1] 1

$saltires
[1] 1

$quarters
[1] 1

$sunstars
[1] 0

$crescent
[1] 1

$triangle
[1] 1

$icon
[1] 0

$animate
[1] 1

$text
[1] 1

$topleft
[1] "red"

$botright
[1] "red"


| Your dedication is inspiring!

  |===========================================================================================================    |  96%
| The only difference between previous examples and this one is that we are defining and using our own function right in
| the call to lapply(). Our function has no name and disappears as soon as lapply() is done using it. So-called
| 'anonymous functions' can be very useful when one of R's built-in functions isn't an option.

...

  |=============================================================================================================  |  98%
| In this lesson, you learned how to use the powerful lapply() and sapply() functions to apply an operation over the
| elements of a list. In the next lesson, we'll take a look at some close relatives of lapply() and sapply().
                                                                                                    
                                                                                                    | In the last lesson, you learned about the two most fundamental members of R's *apply family of functions: lapply() and
| sapply(). Both take a list as input, apply a function to each element of the list, then combine and return the result.
| lapply() always returns a list, whereas sapply() attempts to simplify the result.

...

  |====                                                                                                           |   4%
| In this lesson, you'll learn how to use vapply() and tapply(), each of which serves a very specific purpose within the
                                                                                                    | Split-Apply-Combine methodology. For consistency, we'll use the same dataset we used in the 'lapply and sapply'
| lesson.

...

  |=========                                                                                                      |   8%
| The Flags dataset from the UCI Machine Learning Repository contains details of various nations and their flags. More
| information may be found here: http://archive.ics.uci.edu/ml/datasets/Flags

...

  |=============                                                                                                  |  12%
| I've stored the data in a variable called flags. If it's been a while since you completed the 'lapply and sapply'
| lesson, you may want to reacquaint yourself with the data by using functions like dim(), head(), str(), and summary()
| when you return to the prompt (>). You can also type viewinfo() at the prompt to bring up some documentation for the
| dataset. Let's get started!
                                                                                                      
                                                                                                      ...
                                                                                                    
                                                                                                    |==================                                                                                             |  16%
                                                                                                    | As you saw in the last lesson, the unique() function returns a vector of the unique values contained in the object
                                                                                                    | passed to it. Therefore, sapply(flags, unique) returns a list containing one vector of unique values for each column
                                                                                                    | of the flags dataset. Try it again now.
                                                                                                    
                                                                                                    > 
                                                                                                      > sapply(flags, unique)
                                                                                                    $name
                                                                                                    [1] "Afghanistan"              "Albania"                  "Algeria"                  "American-Samoa"          
                                                                                                    [5] "Andorra"                  "Angola"                   "Anguilla"                 "Antigua-Barbuda"         
                                                                                                    [9] "Argentina"                "Argentine"                "Australia"                "Austria"                 
                                                                                                    [13] "Bahamas"                  "Bahrain"                  "Bangladesh"               "Barbados"                
                                                                                                    [17] "Belgium"                  "Belize"                   "Benin"                    "Bermuda"                 
                                                                                                    [21] "Bhutan"                   "Bolivia"                  "Botswana"                 "Brazil"                  
                                                                                                    [25] "British-Virgin-Isles"     "Brunei"                   "Bulgaria"                 "Burkina"                 
                                                                                                    [29] "Burma"                    "Burundi"                  "Cameroon"                 "Canada"                  
                                                                                                    [33] "Cape-Verde-Islands"       "Cayman-Islands"           "Central-African-Republic" "Chad"                    
                                                                                                    [37] "Chile"                    "China"                    "Colombia"                 "Comorro-Islands"         
                                                                                                    [41] "Congo"                    "Cook-Islands"             "Costa-Rica"               "Cuba"                    
                                                                                                    [45] "Cyprus"                   "Czechoslovakia"           "Denmark"                  "Djibouti"                
                                                                                                    [49] "Dominica"                 "Dominican-Republic"       "Ecuador"                  "Egypt"                   
                                                                                                    [53] "El-Salvador"              "Equatorial-Guinea"        "Ethiopia"                 "Faeroes"                 
                                                                                                    [57] "Falklands-Malvinas"       "Fiji"                     "Finland"                  "France"                  
                                                                                                    [61] "French-Guiana"            "French-Polynesia"         "Gabon"                    "Gambia"                  
                                                                                                    [65] "Germany-DDR"              "Germany-FRG"              "Ghana"                    "Gibraltar"               
                                                                                                    [69] "Greece"                   "Greenland"                "Grenada"                  "Guam"                    
                                                                                                    [73] "Guatemala"                "Guinea"                   "Guinea-Bissau"            "Guyana"                  
                                                                                                    [77] "Haiti"                    "Honduras"                 "Hong-Kong"                "Hungary"                 
                                                                                                    [81] "Iceland"                  "India"                    "Indonesia"                "Iran"                    
                                                                                                    [85] "Iraq"                     "Ireland"                  "Israel"                   "Italy"                   
                                                                                                    [89] "Ivory-Coast"              "Jamaica"                  "Japan"                    "Jordan"                  
                                                                                                    [93] "Kampuchea"                "Kenya"                    "Kiribati"                 "Kuwait"                  
                                                                                                    [97] "Laos"                     "Lebanon"                  "Lesotho"                  "Liberia"                 
                                                                                                    [101] "Libya"                    "Liechtenstein"            "Luxembourg"               "Malagasy"                
                                                                                                    [105] "Malawi"                   "Malaysia"                 "Maldive-Islands"          "Mali"                    
                                                                                                    [109] "Malta"                    "Marianas"                 "Mauritania"               "Mauritius"               
                                                                                                    [113] "Mexico"                   "Micronesia"               "Monaco"                   "Mongolia"                
                                                                                                    [117] "Montserrat"               "Morocco"                  "Mozambique"               "Nauru"                   
                                                                                                    [121] "Nepal"                    "Netherlands"              "Netherlands-Antilles"     "New-Zealand"             
                                                                                                    [125] "Nicaragua"                "Niger"                    "Nigeria"                  "Niue"                    
                                                                                                    [129] "North-Korea"              "North-Yemen"              "Norway"                   "Oman"                    
                                                                                                    [133] "Pakistan"                 "Panama"                   "Papua-New-Guinea"         "Parguay"                 
                                                                                                    [137] "Peru"                     "Philippines"              "Poland"                   "Portugal"                
                                                                                                    [141] "Puerto-Rico"              "Qatar"                    "Romania"                  "Rwanda"                  
                                                                                                    [145] "San-Marino"               "Sao-Tome"                 "Saudi-Arabia"             "Senegal"                 
                                                                                                    [149] "Seychelles"               "Sierra-Leone"             "Singapore"                "Soloman-Islands"         
                                                                                                    [153] "Somalia"                  "South-Africa"             "South-Korea"              "South-Yemen"             
                                                                                                    [157] "Spain"                    "Sri-Lanka"                "St-Helena"                "St-Kitts-Nevis"          
                                                                                                    [161] "St-Lucia"                 "St-Vincent"               "Sudan"                    "Surinam"                 
                                                                                                    [165] "Swaziland"                "Sweden"                   "Switzerland"              "Syria"                   
                                                                                                    [169] "Taiwan"                   "Tanzania"                 "Thailand"                 "Togo"                    
                                                                                                    [173] "Tonga"                    "Trinidad-Tobago"          "Tunisia"                  "Turkey"                  
                                                                                                    [177] "Turks-Cocos-Islands"      "Tuvalu"                   "UAE"                      "Uganda"                  
                                                                                                    [181] "UK"                       "Uruguay"                  "US-Virgin-Isles"          "USA"                     
                                                                                                    [185] "USSR"                     "Vanuatu"                  "Vatican-City"             "Venezuela"               
                                                                                                    [189] "Vietnam"                  "Western-Samoa"            "Yugoslavia"               "Zaire"                   
                                                                                                    [193] "Zambia"                   "Zimbabwe"                
                                                                                                    
                                                                                                    $landmass
                                                                                                    [1] 5 3 4 6 1 2
                                                                                                    
                                                                                                    $zone
                                                                                                    [1] 1 3 2 4
                                                                                                    
                                                                                                    $area
                                                                                                    [1]   648    29  2388     0  1247  2777  7690    84    19     1   143    31    23   113    47  1099   600  8512     6
                                                                                                    [20]   111   274   678    28   474  9976     4   623  1284   757  9561  1139     2   342    51   115     9   128    43
                                                                                                    [39]    22    49   284  1001    21  1222    12    18   337   547    91   268    10   108   249   239   132  2176   109
                                                                                                    [58]   246    36   215   112    93   103  3268  1904  1648   435    70   301   323    11   372    98   181   583   236
                                                                                                    [77]    30  1760     3   587   118   333  1240  1031  1973  1566   447   783   140    41  1267   925   121   195   324
                                                                                                    [96]   212   804    76   463   407  1285   300   313    92   237    26  2150   196    72   637  1221    99   288   505
                                                                                                    [115]    66  2506    63    17   450   185   945   514    57     5   164   781   245   178  9363 22402    15   912   256
                                                                                                    [134]   905   753   391
                                                                                                    
                                                                                                    $population
                                                                                                    [1]   16    3   20    0    7   28   15    8   90   10    1    6  119    9   35    4   24    2   11 1008    5   47   31
                                                                                                    [24]   54   17   61   14  684  157   39   57  118   13   77   12   56   18   84   48   36   22   29   38   49   45  231
                                                                                                    [47]  274   60
                                                                                                    
                                                                                                    $language
                                                                                                    [1] 10  6  8  1  2  4  3  5  7  9
                                                                                                    
                                                                                                    $religion
                                                                                                    [1] 2 6 1 0 5 3 4 7
                                                                                                    
                                                                                                    $bars
                                                                                                    [1] 0 2 3 1 5
                                                                                                    
                                                                                                    $stripes
                                                                                                    [1]  3  0  2  1  5  9 11 14  4  6 13  7
                                                                                                    
                                                                                                    $colours
                                                                                                    [1] 5 3 2 8 6 4 7 1
                                                                                                    
                                                                                                    $red
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $green
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $blue
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $gold
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $white
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $black
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $orange
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $mainhue
                                                                                                    [1] "green"  "red"    "blue"   "gold"   "white"  "orange" "black"  "brown" 
                                                                                                    
                                                                                                    $circles
                                                                                                    [1] 0 1 4 2
                                                                                                    
                                                                                                    $crosses
                                                                                                    [1] 0 1 2
                                                                                                    
                                                                                                    $saltires
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $quarters
                                                                                                    [1] 0 1 4
                                                                                                    
                                                                                                    $sunstars
                                                                                                    [1]  1  0  6 22 14  3  4  5 15 10  7  2  9 50
                                                                                                    
                                                                                                    $crescent
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $triangle
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $icon
                                                                                                    [1] 1 0
                                                                                                    
                                                                                                    $animate
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $text
                                                                                                    [1] 0 1
                                                                                                    
                                                                                                    $topleft
                                                                                                    [1] "black"  "red"    "green"  "blue"   "white"  "orange" "gold"  
                                                                                                    
                                                                                                    $botright
                                                                                                    [1] "green"  "red"    "white"  "black"  "blue"   "gold"   "orange" "brown" 
                                                                                                    
                                                                                                    
                                                                                                    | You are quite good my friend!
                                                                                                      
                                                                                                      |======================                                                                                         |  20%
                                                                                                    | What if you had forgotten how unique() works and mistakenly thought it returns the *number* of unique values contained
                                                                                                    | in the object passed to it? Then you might have incorrectly expected sapply(flags, unique) to return a numeric vector,
                                                                                                    | since each element of the list returned would contain a single number and sapply() could then simplify the result to a
                                                                                                    | vector.
                                                                                                    
                                                                                                    ...
                                                                                                    
                                                                                                    |===========================                                                                                    |  24%
                                                                                                    | When working interactively (at the prompt), this is not much of a problem, since you see the result immediately and
                                                                                                    | will quickly recognize your mistake. However, when working non-interactively (e.g. writing your own functions), a
                                                                                                    | misunderstanding may go undetected and cause incorrect results later on. Therefore, you may wish to be more careful
                                                                                                    | and that's where vapply() is useful.

...

  |===============================                                                                                |  28%
| Whereas sapply() tries to 'guess' the correct format of the result, vapply() allows you to specify it explicitly. If
| the result doesn't match the format you specify, vapply() will throw an error, causing the operation to stop. This can
                                                                                                    | prevent significant problems in your code that might be caused by getting unexpected return values from sapply().
                                                                                                    
                                                                                                    ...
                                                                                                    
                                                                                                    |====================================                                                                           |  32%
                                                                                                    | Try vapply(flags, unique, numeric(1)), which says that you expect each element of the result to be a numeric vector of
                                                                                                    | length 1. Since this is NOT actually the case, YOU WILL GET AN ERROR. Once you get the error, type ok() to continue to
                                                                                                    | the next question.
                                                                                                    
                                                                                                    > vapply(flags, unique, numeric(1))
                                                                                                    Error in vapply(flags, unique, numeric(1)) : values must be length 1,
                                                                                                    but FUN(X[[1]]) result is length 194
                                                                                                    > ok()
                                                                                                    
                                                                                                    | You're the best!

  |========================================                                                                       |  36%
| Recall from the previous lesson that sapply(flags, class) will return a character vector containing the class of each
| column in the dataset. Try that again now to see the result.

> sapply(flags, class)
       name    landmass        zone        area  population    language    religion        bars     stripes     colours 
"character"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" 
        red       green        blue        gold       white       black      orange     mainhue     circles     crosses 
  "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" "character"   "integer"   "integer" 
   saltires    quarters    sunstars    crescent    triangle        icon     animate        text     topleft    botright 
  "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" "character" "character" 

| That's the answer I was looking for.
                                                                                                    
                                                                                                    |============================================                                                                   |  40%
                                                                                                    | If we wish to be explicit about the format of the result we expect, we can use vapply(flags, class, character(1)). The
                                                                                                    | 'character(1)' argument tells R that we expect the class function to return a character vector of length 1 when
                                                                                                    | applied to EACH column of the flags dataset. Try it now.
                                                                                                    
                                                                                                    > vapply(flags, class, character(1))
                                                                                                    name    landmass        zone        area  population    language    religion        bars     stripes     colours 
                                                                                                    "character"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" 
                                                                                                    red       green        blue        gold       white       black      orange     mainhue     circles     crosses 
                                                                                                    "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" "character"   "integer"   "integer" 
                                                                                                    saltires    quarters    sunstars    crescent    triangle        icon     animate        text     topleft    botright 
                                                                                                    "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer"   "integer" "character" "character" 
                                                                                                    
                                                                                                    | Keep working like that and you'll get there!

  |=================================================                                                              |  44%
| Note that since our expectation was correct (i.e. character(1)), the vapply() result is identical to the sapply()
| result -- a character vector of column classes.

...

  |=====================================================                                                          |  48%
| You might think of vapply() as being 'safer' than sapply(), since it requires you to specify the format of the output
| in advance, instead of just allowing R to 'guess' what you wanted. In addition, vapply() may perform faster than
| sapply() for large datasets. However, when doing data analysis interactively (at the prompt), sapply() saves you some
| typing and will often be good enough.

...

  |==========================================================                                                     |  52%
| As a data analyst, you'll often wish to split your data up into groups based on the value of some variable, then apply
                                                                                                    | a function to the members of each group. The next function we'll look at, tapply(), does exactly that.

...

  |==============================================================                                                 |  56%
| Use ?tapply to pull up the documentation.

> ?tapply

| That's correct!
                                                                                                      
                                                                                                      |===================================================================                                            |  60%
                                                                                                    | The 'landmass' variable in our dataset takes on integer values between 1 and 6, each of which represents a different
                                                                                                    | part of the world. Use table(flags$landmass) to see how many flags/countries fall into each group.
                                                                                                    
                                                                                                    > table(flags$landmass)
                                                                                                    
                                                                                                    1  2  3  4  5  6 
                                                                                                    31 17 35 52 39 20 
                                                                                                    
                                                                                                    | You got it right!
                                                                                                      
                                                                                                      |=======================================================================                                        |  64%
                                                                                                    | The 'animate' variable in our dataset takes the value 1 if a country's flag contains an animate image (e.g. an eagle,
| a tree, a human hand) and 0 otherwise. Use table(flags$animate) to see how many flags contain an animate image.

> table(flags$animate)

  0   1 
155  39 

| All that hard work is paying off!

  |===========================================================================                                    |  68%
| This tells us that 39 flags contain an animate object (animate = 1) and 155 do not (animate = 0).

...

  |================================================================================                               |  72%
| If you take the arithmetic mean of a bunch of 0s and 1s, you get the proportion of 1s. Use tapply(flags$animate,
| flags$landmass, mean) to apply the mean function to the 'animate' variable separately for each of the six landmass
| groups, thus giving us the proportion of flags containing an animate image WITHIN each landmass group.

> tapply(flags$animate, flags$landmass, mean)
        1         2         3         4         5         6 
0.4193548 0.1764706 0.1142857 0.1346154 0.1538462 0.3000000 

| You are doing so well!

  |====================================================================================                           |  76%
| The first landmass group (landmass = 1) corresponds to North America and contains the highest proportion of flags with
| an animate image (0.4194).

...

  |=========================================================================================                      |  80%
| Similarly, we can look at a summary of population values (in round millions) for countries with and without the color
| red on their flag with tapply(flags$population, flags$red, summary).

> tapply(flags$population, flags$red, summary)
$`0`
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   0.00    0.00    3.00   27.63    9.00  684.00 

$`1`
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    0.0     0.0     4.0    22.1    15.0  1008.0 


| Nice work!

  |=============================================================================================                  |  84%
| What is the median population (in millions) for countries *without* the color red on their flag?

1: 22.1
2: 0.0
3: 4.0
4: 27.6
5: 3.0
6: 9.0

Selection: 5

| You are doing so well!

  |==================================================================================================             |  88%
| Lastly, use the same approach to look at a summary of population values for each of the six landmasses.

> 
> tapply(flags$population, flags$landmass)
  [1] 5 3 4 6 3 4 1 1 2 2 6 3 1 5 5 1 3 1 4 1 5 2 4 2 1 5 3 4 5 4 4 1 4 1 4 4 2 5 2 4 4 6 1 1 3 3 3 4 1 1 2 4 1 4 4 3 2 6
 [59] 3 3 2 6 4 4 3 3 4 3 3 1 1 6 1 4 4 2 1 1 5 3 3 5 6 5 5 3 5 3 4 1 5 5 5 4 6 5 5 5 4 4 4 3 3 4 4 5 5 4 3 6 4 4 1 6 3 5
[117] 1 4 4 6 5 3 1 6 1 4 4 6 5 5 3 5 5 2 6 2 2 6 3 3 1 5 3 4 3 4 5 4 4 4 5 6 4 4 5 5 3 5 4 1 1 1 4 2 4 3 3 5 5 4 5 4 6 2
[175] 4 5 1 6 5 4 3 2 1 1 5 6 3 2 5 6 3 4 4 4

| You're close...I can feel it! Try it again. Or, type info() for more options.
                                                                                                    
                                                                                                    | You can see a summary of populations for each of the six landmasses by calling tapply() with three arguments:
                                                                                                      | flags$population, flags$landmass, and summary.
                                                                                                    
                                                                                                    > tapply(flags$population, flags$landmass, summary)
                                                                                                    $`1`
                                                                                                    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                                                    0.00    0.00    0.00   12.29    4.50  231.00 
                                                                                                    
                                                                                                    $`2`
                                                                                                    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                                                    0.00    1.00    6.00   15.71   15.00  119.00 
                                                                                                    
                                                                                                    $`3`
                                                                                                    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                                                    0.00    0.00    8.00   13.86   16.00   61.00 
                                                                                                    
                                                                                                    $`4`
                                                                                                    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                                                    0.000   1.000   5.000   8.788   9.750  56.000 
                                                                                                    
                                                                                                    $`5`
                                                                                                    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                                                    0.00    2.00   10.00   69.18   39.00 1008.00 
                                                                                                    
                                                                                                    $`6`
                                                                                                    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                                                    0.00    0.00    0.00   11.30    1.25  157.00 
                                                                                                    
                                                                                                    
                                                                                                    | That's a job well done!

  |======================================================================================================         |  92%
| What is the maximum population (in millions) for the fourth landmass group (Africa)?

1: 1010.0
2: 157.00
3: 56.00
4: 119.0
5: 5.00

Selection: 3

| Excellent work!

  |===========================================================================================================    |  96%
| In this lesson, you learned how to use vapply() as a safer alternative to sapply(), which is most helpful when writing
| your own functions. You also learned how to use tapply() to split your data into groups based on the value of some
| variable, then apply a function to each group. These functions will come in handy on your quest to become a better
| data analyst.

database check

class()
dim()
nrow
ncol
object.size()
names() - vector column names

| By default, head() shows you the first six rows of the data. You can alter this behavior by passing as a second
| argument the number of rows you'd like to view. Use head() to preview the first 10 rows of plants.
                                                                                                    
| The same applies for using tail() to preview the end of the dataset. Use tail() to view the last 15 rows.
summary()                          
table()

| Perhaps the most useful and concise function for understanding the *str*ucture of your data is str(). Give it a try
| now.

> str()
Error in str.default() : argument "object" is missing, with no default
> str(plants)
'data.frame':	5166 obs. of  10 variables:
  $ Scientific_Name     : chr  "Abelmoschus" "Abelmoschus esculentus" "Abies" "Abies balsamea" ...
$ Duration            : chr  NA "Annual, Perennial" NA "Perennial" ...
$ Active_Growth_Period: chr  NA NA NA "Spring and Summer" ...
$ Foliage_Color       : chr  NA NA NA "Green" ...
$ pH_Min              : num  NA NA NA 4 NA NA NA NA 7 NA ...
$ pH_Max              : num  NA NA NA 6 NA NA NA NA 8.5 NA ...
$ Precip_Min          : int  NA NA NA 13 NA NA NA NA 4 NA ...
$ Precip_Max          : int  NA NA NA 60 NA NA NA NA 20 NA ...
$ Shade_Tolerance     : chr  NA NA NA "Tolerant" ...
$ Temp_Min_F          : int  NA NA NA -43 NA NA NA NA -13 NA ...

| Perseverance, that's the answer.

  |==================================================================================================             |  88%
| The beauty of str() is that it combines many of the features of the other functions you've already seen, all in a
| concise and readable format. At the very top, it tells us that the class of plants is 'data.frame' and that it has
| 5166 observations and 10 variables. It then gives us the name and class of each variable, as well as a preview of its
| contents.

...

|======================================================================================================         |  92%
| str() is actually a very general function that you can use on most objects in R. Any time you want to understand the
| structure of something (a dataset, function, etc.), str() is a good place to start.

...

|===========================================================================================================    |  96%
| In this lesson, you learned how to get a feel for the structure and contents of a new dataset using a collection of
| simple and useful functions. Taking the time to do this upfront can save you time and frustration later on in your
| analysis.
]

| Let's simulate rolling four six-sided dice: sample(1:6, 4, replace = TRUE).

> sample(1:6, 4, replace = TRUE)

| Now sample 10 numbers between 1 and 20, WITHOUT replacement. To sample without replacement, simply leave off the
| 'replace' argument.

 sample(1:20, 10)
 
 LETTERS is a predefined variable in R containing a vector of all 26 letters of the English alphabet
 
 | The sample() function can also be used to permute, or rearrange, the elements of a vector. For example, try
| sample(LETTERS) to permute all 26 letters of the English alphabet.

| Now, suppose we want to simulate 100 flips of an unfair two-sided coin. This particular coin has a 0.3 probability of
| landing 'tails' and a 0.7 probability of landing 'heads'.

...

  |========================================                                                                       |  36%
| Let the value 0 represent tails and the value 1 represent heads. Use sample() to draw a sample of size 100 from the
| vector c(0,1), with replacement. Since the coin is unfair, we must attach specific probabilities to the values 0
| (tails) and 1 (heads) with a fourth argument, prob = c(0.3, 0.7). Assign the result to a new variable called flips.

> 
> flips <- sample(c(0, 1), 100, replace = TRUE,prob=c(0.3,0.7))

| You nailed it! Good job!

  |============================================                                                                   |  39%
| View the contents of the flips variable.

> flips
  [1] 0 1 0 0 1 1 1 0 0 1 1 0 1 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1 0 1 0 1 1 1 0 0 1 0 0 0 1 1 0 1
 [59] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 0 1 1 1 1 1 0 1 0 0 0 0 1 0

| Excellent job!

  |===============================================                                                                |  42%
| Since we set the probability of landing heads on any given flip to be 0.7, we'd expect approximately 70 of our coin
| flips to have the value 1. Count the actual number of 1s contained in flips using the sum() function.

> sum(flips)
[1] 68

|==================================================                                                             |  45%
| A coin flip is a binary outcome (0 or 1) and we are performing 100 independent trials (coin flips), so we can use
| rbinom() to simulate a binomial random variable. Pull up the documentation for rbinom() using ?rbinom.

> ?rbinom

| You are quite good my friend!
  
  |======================================================                                                         |  48%
| Each probability distribution in R has an r*** function (for "random"), a d*** function (for "density"), a p*** (for
                                                                                                                   | "probability"), and q*** (for "quantile"). We are most interested in the r*** functions in this lesson, but I
| encourage you to explore the others on your own.

...

|=========================================================                                                      |  52%
| A binomial random variable represents the number of 'successes' (heads) in a given number of independent 'trials'
| (coin flips). Therefore, we can generate a single random variable that represents the number of heads in 100 flips of
| our unfair coin using rbinom(1, size = 100, prob = 0.7). Note that you only specify the probability of 'success'
| (heads) and NOT the probability of 'failure' (tails). Try it now.

> rbinom(flips)
Error in rbinom(flips) : argument "size" is missing, with no default
> rbinom(1, size=100, prob=0.7)
[1] 82

| Great job!
  
  |=============================================================                                                  |  55%
| Equivalently, if we want to see all of the 0s and 1s, we can request 100 observations, each of size 1, with success
| probability of 0.7. Give it a try, assigning the result to a new variable called flips2.

> flips <- rbinom(1, size=100, prob=0.7)

| Try again. Getting it right on the first try is boring anyway! Or, type info() for more options.

| Call rbinom() with n = 100, size = 1, and prob = 0.7 and assign the result to flips2.

> flips <- rbinom(100, size=100, prob=0.7)

| Almost! Try again. Or, type info() for more options.

| Call rbinom() with n = 100, size = 1, and prob = 0.7 and assign the result to flips2.

> flips <- rbinom(100, size=1, prob=0.7)

| You're close...I can feel it! Try it again. Or, type info() for more options.

| Call rbinom() with n = 100, size = 1, and prob = 0.7 and assign the result to flips2.

> flips <- rbinom(n=100, size=1, prob=0.7)

| Give it another try. Or, type info() for more options.

| Call rbinom() with n = 100, size = 1, and prob = 0.7 and assign the result to flips2.

> flips2 <- rbinom(n=100, size=1, prob=0.7)

| Great job!

  |================================================================                                               |  58%
| View the contents of flips2.

> flips2
  [1] 0 0 0 1 0 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1 0 1 1 0 0 1 0 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1
 [59] 1 1 1 0 1 0 1 0 1 0 1 0 1 1 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 0 1 1 0 0

| All that practice is paying off!

  |===================================================================                                            |  61%
| Now use sum() to count the number of 1s (heads) in flips2. It should be close to 70!

> sum(flips2)
[1] 66

| Excellent job!

  |=======================================================================                                        |  64%
| Similar to rbinom(), we can use R to simulate random numbers from many other probability distributions. Pull up the
| documentation for rnorm() now.

> ?rnorm

| You're the best!
  
  |==========================================================================                                     |  67%
| The standard normal distribution has mean 0 and standard deviation 1. As you can see under the 'Usage' section in the
| documentation, the default values for the 'mean' and 'sd' arguments to rnorm() are 0 and 1, respectively. Thus,
| rnorm(10) will generate 10 random numbers from a standard normal distribution. Give it a try.

> rnomr(10)
Error in rnomr(10) : could not find function "rnomr"
> rnorm(10)
[1]  2.7605091  0.8771307 -0.1152868 -0.8362446 -1.0931190 -0.2592897  0.1267114 -0.1412626 -0.4406887  0.5856328

| Nice work!
  
  |=============================================================================                                  |  70%
| Now do the same, except with a mean of 100 and a standard deviation of 25.

> rnorm(10, mean=100, sd=25)
[1] 144.56926 121.02147  98.92706 125.01073 122.44560  92.57272  99.00600 110.64762  71.80015  60.24806

| You are amazing!
  
  |=================================================================================                              |  73%
| Finally, what if we want to simulate 100 *groups* of random numbers, each containing 5 values generated from a Poisson
| distribution with mean 10? Let's start with one group of 5 numbers, then I'll show you how to repeat the operation 100
| times in a convenient and compact way.

...

|====================================================================================                           |  76%
| Generate 5 random values from a Poisson distribution with mean 10. Check out the documentation for rpois() if you need
| help.

> ?rpois
> rpois(n=5, 10)
[1] 14 13 13  5  9

| That's correct!

  |=======================================================================================                        |  79%
| Now use replicate(100, rpois(5, 10)) to perform this operation 100 times. Store the result in a new variable called
| my_pois.

> replicate(100, rpois(5, 10))
     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20]
[1,]    5   15    9   12    8    6   11   10   12    12     9    10    11     8    14     9     9    11     9     8
[2,]   10    7   16    4   12   11   12    4    7     7    10     7    12     4    12     7    10    16    13     7
[3,]    8   10   13   13    9    8   13   15    5     6     9    11    14    14     9    11     7    14     7     6
[4,]   12   11   15    8   10   12    8    6    7    11     8     6     5    12     9     8     7     9     7    13
[5,]    5    9   13    6   14    9    5    6   13    12    12     7     9    10    11     6     9    19    10    10
     [,21] [,22] [,23] [,24] [,25] [,26] [,27] [,28] [,29] [,30] [,31] [,32] [,33] [,34] [,35] [,36] [,37] [,38] [,39]
[1,]     7     8     9     7     9     8    16     7    14    10    17    11    10     8    12     9     9    13    13
[2,]    17    17     7    19    14     5    15    12    10    13    12    11    12     8     9    14    12     7    10
[3,]     6    14     7     9    12    10    12    14    11    17    13    12    14    14     9    10     9    11     7
[4,]     5    10     9    13     8     9    11    12     9    11    12     8    10    11    13     9    12     5    14
[5,]     4    13     9     7     6     8     8     8     4    10    11     9    14    20     8     7    10     7     8
     [,40] [,41] [,42] [,43] [,44] [,45] [,46] [,47] [,48] [,49] [,50] [,51] [,52] [,53] [,54] [,55] [,56] [,57] [,58]
[1,]    11    12     9    10    10     6    18    11     6    14     7    16     9    16    10    14     8    13     4
[2,]    12     8     7     9    11     7    10    13     9     9     3     9     9    15    12     9     9    13     8
[3,]    11    17    10     9    12    12    12    14    11     9     9     9     8     8    10     9    11     3    19
[4,]    13     5    13    11     8    12     8     5     9    18    11    10     6    11    18     9    10    13    16
[5,]     9     7     6    10     9    14     6    13     3     9    15     8    18     6     7    17    13     8     6
     [,59] [,60] [,61] [,62] [,63] [,64] [,65] [,66] [,67] [,68] [,69] [,70] [,71] [,72] [,73] [,74] [,75] [,76] [,77]
[1,]    10    15     7    14    14    11     9    11    12     9     9    18    18    15    16    12    13    10     8
[2,]     9    11    10    10     9     4    12     8    11     9     9     8    10    10     4     7     4     8     9
[3,]    15     6     5     7     7     7    12     8     7    14     8    10    13    10    12    12    10     8     7
[4,]    10    10    14    15    12     9    16     9    10     9     8    10    18    14    18     5     5     9     6
[5,]    10    12    13    11    11     9    10    16     9     6     6     8     5     7    12     6     7    10     8
     [,78] [,79] [,80] [,81] [,82] [,83] [,84] [,85] [,86] [,87] [,88] [,89] [,90] [,91] [,92] [,93] [,94] [,95] [,96]
[1,]    11     7    12    15    13    10    15    14     7    19    12    15     7     7    14     5     9     7    11
[2,]     5    15    18    13     9    12    15    10    11    12     6     6     6    13     6    13    12    15    10
[3,]    10     4     9     7    11    11    11    11     3    13    10    14     9    10    10     5     7    15    18
[4,]     7     9    11    13    15     3     8    12    13     8    12    10     6    11    11    10     8     9    14
[5,]     7    10     9     8    11     5     9    11     7     9     8     7    12     8     5     7     6     7    10
     [,97] [,98] [,99] [,100]
[1,]    12    11     6     10
[2,]    15    10    11      7
[3,]    10     7     9      9
[4,]     8     9     9      7
[5,]    11    16    16     13

| You're close...I can feel it! Try it again. Or, type info() for more options.

| my_pois <- replicate(100, rpois(5, 10)) will repeat the operation 100 times and store the result.

> my_pois <- replicate(100, rpois(5, 10))

| Excellent job!
  
  |===========================================================================================                    |  82%
| Take a look at the contents of my_pois.

> my_pois
[,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20]
[1,]    7   13    7    7    8    8    8    4   11     7     3    18    11    10    10     8    15    10     8    12
[2,]   19   14    6   12   13   12   12   11   12    18    10    10     9     5     7    11     3     7    12     6
[3,]    6    7    8   18   12   15   11    5    6     7     8     4    11     8    11     7     8     8     9     7
[4,]   11    9   14   15   13   11   12   12   14    14    12     8    11     6     7     6    16     7     7     8
[5,]    8    7    9   10   14    7    6   18   17    11     7    13     9    15    11    15    10     4    14    10
[,21] [,22] [,23] [,24] [,25] [,26] [,27] [,28] [,29] [,30] [,31] [,32] [,33] [,34] [,35] [,36] [,37] [,38] [,39]
[1,]    12     5     8    10     7    12     8     5     6     9    12     7    12     8    14    12     3     7     8
[2,]    10    11     9     7     8     9    13     7    13    11     8     8    11     8    21     7     4    11     8
[3,]     6    10    13    11    11     5     9    10     9    13    10    15    11     9     7     7     6    11     8
[4,]     8    17    11    11     9    15     7     5     9     6     8     4     9    15    12     5    11    13    12
[5,]    17     7     9     8    12     6    17    13    10    12     9    10     9    10     4    12    15    11     5
[,40] [,41] [,42] [,43] [,44] [,45] [,46] [,47] [,48] [,49] [,50] [,51] [,52] [,53] [,54] [,55] [,56] [,57] [,58]
[1,]     7     8     4     7    12     8    17    15    12     8    18    11    11     6    11    15    10     9     8
[2,]    13    13    11     8    12    11    14    16    10    10    10    14    13     9    11     6    13     7    12
[3,]    13     5    13     9     5    12     8     9     9     7    10    14     9     9     7    18    12     8     9
[4,]    12     7    14    12    17     8    12     8     8    15    13     2     7     7    15    15     7     8    15
[5,]    12     9     9     5    11     5    12     7    10    11     9    14     8    12     6    11    10    10    10
[,59] [,60] [,61] [,62] [,63] [,64] [,65] [,66] [,67] [,68] [,69] [,70] [,71] [,72] [,73] [,74] [,75] [,76] [,77]
[1,]     8     6    13    10     7    15    12    12     7    10    11    10    11    14    11     6    12    11    12
[2,]    11    13    11    12     9    14    14    12     5    10    16     8    11    19    11    12     7     5    12
[3,]    14     9     7     6    10    12    10    11     8     8     2     7     7    13     7    12    11    10    11
[4,]     9    10     7    10    18     4    10    19    15     8    12    12     9    11    11    12    10    10    14
[5,]    20    13     8     7     9     8    12    10     9    10    13    17    12    10    12    13    11    10     8
[,78] [,79] [,80] [,81] [,82] [,83] [,84] [,85] [,86] [,87] [,88] [,89] [,90] [,91] [,92] [,93] [,94] [,95] [,96]
[1,]    11    16     9    10     7     8     9    10     9     4    11    12    11     8     6     6     8    10    18
[2,]    12    16    14    11    10     9     2     8     9     8    14    15    10    10    12    10     4     9     7
[3,]    15    14     9    18    11    11    12     5     8     8    14    14     5     9    15     9     9     6     7
[4,]    13    13     8    10    12     9     8    10     3    17    12    10    14    12    13     7     6     9    16
[5,]    10    10     7     9    12     9    12     7    11    10    10    11    10    10     5    13    12    12    12
[,97] [,98] [,99] [,100]
[1,]     6     6     7     15
[2,]     7     8    16     17
[3,]    16    10    12     11
[4,]     6     7     8     10
[5,]     7     6     6     12

| That's correct!

  |==============================================================================================                 |  85%
| replicate() created a matrix, each column of which contains 5 random numbers generated from a Poisson distribution
| with mean 10. Now we can find the mean of each column in my_pois using the colMeans() function. Store the result in a
| variable called cm.

> cm <- colMeans(my_pois)

| Great job!

  |==================================================================================================             |  88%
| And let's take a look at the distribution of our column means by plotting a histogram with hist(cm).

> hist(cm)

| You are really on a roll!
  
  |=====================================================================================================          |  91%
| Looks like our column means are almost normally distributed, right? That's the Central Limit Theorem at work, but
| that's a lesson for another day!
  
  ...

|========================================================================================================       |  94%
| All of the standard probability distributions are built into R, including exponential (rexp()), chi-squared
| (rchisq()), gamma (rgamma()), .... Well, you see the pattern.

...

|============================================================================================================   |  97%
| Simulation is practically a field of its own and we've only skimmed the surface of what's possible. I encourage you to
| explore these and other functions further on your own.


|   6%
| Let's start by using d1 <- Sys.Date() to get the current date and store it in the variable d1. (That's the letter 'd'
| and the number 1.)

> d1 <- Sys.Date()

| That's a job well done!

  |=========                                                                                                      |   8%
| Use the class() function to confirm d1 is a Date object.

> class(d1)
[1] "Date"

| Great job!

  |============                                                                                                   |  11%
| We can use the unclass() function to see what d1 looks like internally. Try it out.

> unclass(d1)
[1] 19229

> t1 <- Sys.time()

| All that hard work is paying off!

  |==================================                                                                             |  31%
| View the contents of t1.

> t1
[1] "2022-08-25 03:08:24 CDT"

> class(t1)
[1] "POSIXct" "POSIXt" 
  |========================================                                                                       |  36%
| As mentioned earlier, POSIXct is just one of two ways that R represents time information. (You can ignore the second
| value above, POSIXt, which just functions as a common language between POSIXct and POSIXlt.) Use unclass() to see what
| t1 looks like internally -- the (large) number of seconds since the beginning of 1970.

> unclass(t1)
[1] 1661414905

| You're the best!
  
  |===========================================                                                                    |  39%
| By default, Sys.time() returns an object of class POSIXct, but we can coerce the result to POSIXlt with
| as.POSIXlt(Sys.time()). Give it a try and store the result in t2.

> t2 <- POSIXlt(Sys.time())
Error in POSIXlt(Sys.time()) : could not find function "POSIXlt"
> t2 <- as.POSIXlt(Sys.time())

| You got it!
  
  |==============================================                                                                 |  42%
| Check the class of t2.

> clasS(t2)
Error in clasS(t2) : could not find function "clasS"
> clasS(2)
Error in clasS(2) : could not find function "clasS"
> class(2)
[1] "numeric"

| You almost had it, but not quite. Try again. Or, type info() for more options.

| Type class(t2) to view its class.

> class(2)
[1] "numeric"

| Not exactly. Give it another go. Or, type info() for more options.

| Type class(t2) to view its class.

> class(t2)
[1] "POSIXlt" "POSIXt" 

| Excellent job!
  
  |=================================================                                                              |  44%
| Now view its contents.

> t2
[1] "2022-08-25 03:09:39 CDT"

| Excellent work!
  
  |====================================================                                                           |  47%
| The printed format of t2 is identical to that of t1. Now unclass() t2 to see how it is different internally.

> unclass(t2)
$sec
[1] 39.27212

$min
[1] 9

$hour
[1] 3

$mday
[1] 25

$mon
[1] 7

$year
[1] 122

$wday
[1] 4

$yday
[1] 236

$isdst
[1] 1

$zone
[1] "CDT"

$gmtoff
[1] -18000

attr(,"tzone")
[1] ""    "CST" "CDT"

| That's correct!

  |========================================================                                                       |  50%
| t2, like all POSIXlt objects, is just a list of values that make up the date and time. Use str(unclass(t2)) to have a
| more compact view.

> str(unclass(t2))
List of 11
 $ sec   : num 39.3
 $ min   : int 9
 $ hour  : int 3
 $ mday  : int 25
 $ mon   : int 7
 $ year  : int 122
 $ wday  : int 4
 $ yday  : int 236
 $ isdst : int 1
 $ zone  : chr "CDT"
 $ gmtoff: int -18000
 - attr(*, "tzone")= chr [1:3] "" "CST" "CDT"

| That's a job well done!
  
  |===========================================================                                                    |  53%
| If, for example, we want just the minutes from the time stored in t2, we can access them with t2$min. Give it a try.

> t2$min
[1] 9

| You nailed it! Good job!
  
  |==============================================================                                                 |  56%
| Now that we have explored all three types of date and time objects, let's look at a few functions that extract useful
| information from any of these objects -- weekdays(), months(), and quarters().

...

  |=================================================================                                              |  58%
| The weekdays() function will return the day of week from any date or time object. Try it out on d1, which is the Date
| object that contains today's date.

> weekdays()
Error in UseMethod("weekdays") : 
  no applicable method for 'weekdays' applied to an object of class "NULL"
> weekdays(d1)
[1] "Thursday"

| You're the best!

  |====================================================================                                           |  61%
| The months() function also works on any date or time object. Try it on t1, which is the POSIXct object that contains
| the current time (well, it was the current time when you created it).

> months(d1)
[1] "August"

| That's not exactly what I'm looking for. Try again. Or, type info() for more options.

| months(t1) will give you the current month.

> months(t1)
[1] "August"

| You're the best!
  
  |=======================================================================                                        |  64%
| The quarters() function returns the quarter of the year (Q1-Q4) from any date or time object. Try it on t2, which is
| the POSIXlt object that contains the time at which you created it.

> quarters(t2)
[1] "Q3"

| Keep up the great work!
  
  
  |=============================================================================                                  |  69%
| strptime() converts character vectors to POSIXlt. In that sense, it is similar to as.POSIXlt(), except that the input
| doesn't have to be in a particular format (YYYY-MM-DD).

...strptime()

  |================================================================================                               |  72%
| To see how it works, store the following character string in a variable called t3: "October 17, 1986 08:24" (with the
| quotes).

> t3 <- "October 17, 1986 08:24"

| Excellent work!

  |===================================================================================                            |  75%
| Now, use strptime(t3, "%B %d, %Y %H:%M") to help R convert our date/time object to a format that it understands.
| Assign the result to a new variable called t4. (You should pull up the documentation for strptime() if you'd like to
| know more about how it works.)

> t4 <- strptime(t3, "%B %d, %Y %H:%M")

| Keep up the great work!
  
  |======================================================================================                         |  78%
| Print the contents of t4.

> t4
[1] "1986-10-17 08:24:00 CDT"

| The variable t1 contains the time at which you created it (recall you used Sys.time()). Confirm that some time has
| passed since you created t1 by using the 'greater than' operator to compare it to the current time: Sys.time() > t1

> sys.time() > t1
Error in sys.time() : could not find function "sys.time"
> Sys.time() > t1
[1] TRUE

| Your dedication is inspiring!
  
  |===================================================================================================            |  89%
| So we know that some time has passed, but how much? Try subtracting t1 from the current time using Sys.time() - t1.
| Don't forget the parentheses at the end of Sys.time(), since it is a function.

> Sys.time() - t1
Time difference of 4.949958 mins

| You are really on a roll!

  |======================================================================================================         |  92%
| The same line of thinking applies to addition and the other comparison operators. If you want more control over the
| units when finding the above difference in times, you can use difftime(), which allows you to specify a 'units'
| parameter.

...

  |=========================================================================================================      |  94%
| Use difftime(Sys.time(), t1, units = 'days') to find the amount of time in DAYS that has passed since you created t1.

> difftime(Sys.time(), t1, units='days')
Time difference of 0.00372343 days

  



 [1] 16 12 17 20  9  8 11  6  4 18
