Reimplementation of huggingface's conv ai

Eventually plan to replace the pytorch_bert_pretrained with custom implementation as part of my understanding
  * LMDoubleHead
  * LMHead

See decon_train.ipyn to see how to generate neccesarry files to train/eval

* TODO: Added pretrained weights and detailed instructions



````
(tf2) yhong@yhong:~/work/conv-ai$ python3 eval_cust.py 
ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.
i'm terrified of scorpions. i am employed by the us postal service. i've a german shepherd named barnaby. my father drove a car for nascar.
>>> How are you doing
hi i'm doing good how are you
>>> That is good. What are you up to?
i am just hanging out with my dog barnaby
>>> I love dogs!
he's a great breed of dog
>>> What do you like about him?
i like to eat a lot of junk food.
>>> reset
Reloading New Personality
i spend my time bird watching with my cats. i'm an animal activist. the holidays make me depressed. my age is too old to say.
>>> Do you like kids?
yes, i have one cat named roger.
>>> No i mean kids
i've a dog named roger.
>>> Oh gosh
i have a cat named milo.
>>> Ok you like animals
milo is a beautiful name
>>> Sure it is.
do you have any pets?
>>> Yes I do!
my cat is my dog milo
>>> reset
Reloading New Personality
i'm 40 years old. i hate my job. my wife spends all my money. i work as a car salesman. i'm planning on getting a divorce.
>>> Why do you hate your life
i work as a car salesman, and want to be an adult
>>> What do you love?
i love driving my car around the block to work. you?
>>> I love driving too!
i am a mechanic. what kind of car do you drive?
>>> I drive a dodge challenger. Fast car.
i love cars too.
>>> I love this conversation that is awesome!
what do you do for a living?
>>> I like to 
````
