# 2-Transformers, Explained: Understand the Model Behind GPT-3, BERT, and T5 --- 变形金刚，解释：了解 GPT-3、BERT 和 T5 背后的模型
[Transformers, Explained: Understand the Model Behind GPT-3, BERT, and T5 --- 变形金刚，解释：了解 GPT-3、BERT 和 T5 背后的模型](https://daleonai.com/transformers-explained) 

 You know that expression _When you have a hammer, everything looks like a nail_? Well, in machine learning, it seems like we really have discovered a magical hammer for which everything is, in fact, a nail, and they’re called Transformers. Transformers are models that can be designed to translate text, write [poems and op eds](https://www.gwern.net/GPT-3), and [even generate computer code](https://www.wired.com/story/ai-latest-trick-writing-computer-code/). In fact, lots of the amazing research I write about on daleonai.com is built on Transformers, like [AlphaFold 2](https://daleonai.com/how-alphafold-works), the model that predicts the structures of proteins from their genetic sequences, as well as powerful natural language processing (NLP) models like [GPT-3](https://daleonai.com/how-alphafold-works), BERT, T5, Switch, Meena, and others. You might say they’re more than meets the… ugh, forget it.  
你知道那句话当你拿着锤子时，一切看起来都像钉子吗？好吧，在机器学习中，我们似乎真的发现了一把神奇的锤子，实际上一切都是钉子，它们被称为变形金刚。变形金刚是可以设计用来翻译文本、写诗和专栏，甚至生成计算机代码的模型。事实上，我在 daleonai.com 上写的许多令人惊叹的研究都是建立在变形金刚之上的，例如 AlphaFold 2，该模型可以根据基因序列预测蛋白质的结构，以及强大的自然语言处理 (NLP) 模型，例如 GPT -3、BERT、T5、Switch、Meena 等。你可能会说他们不仅仅是满足...... 呃，算了吧。

If you want to stay hip in machine learning and especially NLP, you have to know at least a bit about Transformers. So in this post, we’ll talk about what they are, how they work, and why they’ve been so impactful.  
如果你想在机器学习尤其是 NLP 领域保持领先，你必须至少对 Transformers 有所了解。所以在这篇文章中，我们将讨论它们是什么、它们如何工作以及它们为何如此有影响力。

* * *

A Transformer is a type of neural network architecture. To recap, neural nets are a very effective type of model for analyzing complex data types like images, videos, audio, and text. But there are different types of neural networks optimized for different types of data.  
Transformer 是一种神经网络架构。回顾一下，神经网络是一种非常有效的模型，可用于分析图像、视频、音频和文本等复杂数据类型。但是有不同类型的神经网络针对不同类型的数据进行了优化。  
For example, for analyzing images, we’ll typically use [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) or “CNNs.” Vaguely, they mimic the way the human brain processes visual information.  
例如，为了分析图像，我们通常会使用卷积神经网络或 “CNN”。它们模糊地模仿了人脑处理视觉信息的方式。

![](https://daleonai.com/images/cnn.png)

_Convolutional Neural Network, courtesy Renanar2 at Wikicommons.  
卷积神经网络，由 Wikicommons 的 Renanar2 友情提供。_

And [since around 2012](https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/), we’ve been quite successful at solving vision problems with CNNs, like identifying objects in photos, recognizing faces, and reading handwritten digits.  
自 2012 年左右以来，我们在使用 CNN 解决视觉问题方面取得了相当大的成功，例如识别照片中的对象、识别人脸和阅读手写数字。  
But for a long time, nothing comparably good existed for language tasks (translation, text summarization, text generation, named entity recognition, etc). That was unfortunate, because language is the main way we humans communicate.  
但长期以来，语言任务（翻译、文本摘要、文本生成、命名实体识别等）都没有比这更好的了。这很不幸，因为语言是我们人类交流的主要方式。

Before Transformers were introduced in 2017, the way we used deep learning to understand text was with a type of model called a Recurrent Neural Network or RNN that looked something like this:  
在 2017 年推出 Transformers 之前，我们使用深度学习来理解文本的方式是使用一种称为循环神经网络或 RNN 的模型，它看起来像这样：

![](https://daleonai.com/images/renn.png)

_Image of an RNN, courtesy Wikimedia.  
RNN 图片，由维基媒体提供。_

Let’s say you wanted to translate a sentence from English to French. An RNN would take as input an English sentence, process the words one at a time, and then, sequentially, spit out their French counterparts. The key word here is “sequential.  
假设您想将一个句子从英语翻译成法语。循环神经网络会将英语句子作为输入，一次处理一个单词，然后依次吐出对应的法语单词。这里的关键词是 “顺序的”。  
” In language, the order of words matters and you can’t just shuffle them around. The sentence:  
” 在语言中，单词的顺序很重要，你不能随意乱序。这句话：

“Jane went looking for trouble.”“简去自找麻烦了。”

means something very different from the sentence:  
意味着与句子截然不同的东西：

“Trouble went looking for Jane”“麻烦去找简”

So any model that’s going to understand language must capture word order, and recurrent neural networks did this by processing one word at a time, in a sequence.  
因此，任何要理解语言的模型都必须捕捉词序，而循环神经网络通过按顺序一次处理一个词来做到这一点。

But RNNs had issues. First, they struggled to handle large sequences of text, like long paragraphs or essays. By the time got to the end of a paragraph, they’d forget what happened at the beginning.  
但是 RNN 有问题。首先，他们很难处理大量文本序列，例如长段落或文章。到段落结尾时，他们会忘记开头发生的事情。  
An RNN-based translation model, for example, might have trouble remembering the gender of the subject of a long paragraph.  
例如，基于 RNN 的翻译模型可能难以记住长段落主题的性别。

Worse, RNNs were hard to train. They were notoriously susceptible to what’s called the [vanishing/exploding gradient problem](https://towardsdatascience.com/the-exploding-and-vanishing-gradients-problem-in-time-series-6b87d558d22) (sometimes you simply had to restart training and cross your fingers). Even more problematic, because they processed words sequentially, RNNs were hard to parallelize.  
更糟糕的是，RNN 很难训练。众所周知，它们很容易受到所谓的消失 / 爆炸梯度问题的影响（有时你只需要重新开始训练并祈祷）。更有问题的是，因为它们按顺序处理单词，RNN 很难并行化。  
This meant you couldn’t just speed up training by throwing more GPUs at the them, which meant, in turn, you couldn’t train them on all that much data.  
这意味着您不能仅仅通过向它们投放更多 GPU 来加快训练速度，这反过来又意味着您无法在所有那么多数据上训练它们。

## Enter Transformers 进入变形金刚

This is where Transformers changed everything. They were developed in 2017 by researchers at Google and the University of Toronto, initially designed to do translation. But unlike recurrent neural networks, Transformers could be very efficiently parallelized.  
这就是变形金刚改变一切的地方。它们由谷歌和多伦多大学的研究人员于 2017 年开发，最初旨在进行翻译。但与循环神经网络不同，Transformer 可以非常高效地并行化。  
And that meant, with the right hardware, you could train some really big models.  
这意味着，使用合适的硬件，你可以训练一些非常大的模型。

How big? 多大？

Bigly big. 很大很大。

GPT-3, the especially impressive text-generation model that writes almost as well as a human was trained on some _45 TB_ of text data, including almost all of the public web.  
GPT-3 是一种特别令人印象深刻的文本生成模型，它的书写能力几乎与人类一样好，它在大约 45 TB 的文本数据上进行了训练，包括几乎所有的公共网络。

So if you remember anything about Transformers, let it be this: combine a model that scales well with a huge dataset and the results will likely blow you away.  
因此，如果您还记得有关变形金刚的任何事情，那就是：将一个可很好扩展的模型与一个巨大的数据集结合起来，结果可能会让您大吃一惊。

## How do Transformers Work? 变压器如何工作？

![](https://daleonai.com/images/screen-shot-2021-05-06-at-12.12.21-pm.png)

_Transformer diagram from the original paper  
原始论文中的变压器图_

While the diagram from the [original paper](https://arxiv.org/abs/1706.03762) is a little scary, the innovation behind Transformers boils down to three main concepts:  
虽然原始论文中的图表有点吓人，但变形金刚背后的创新归结为三个主要概念：

1.  Positional Encodings 位置编码
2.  Attention 注意力
3.  Self-Attention 自注意力

#### Positional Encodings 位置编码

Let’s start with the first one, positional encodings. Let’s say we’re trying to translate text from English to French. Remember that RNNs, the old way of doing translation, understood word order by processing words sequentially. But this is also what made them hard to parallelize.  
让我们从第一个开始，位置编码。假设我们正在尝试将文本从英语翻译成法语。请记住，RNN 是一种旧的翻译方式，它通过按顺序处理单词来理解单词顺序。但这也是使它们难以并行化的原因。  

Transformers get around this barrier via an innovational called positional encodings. The idea is to take all of the words in your input sequence–an English sentence, in this case–and append each word with a number it’s order. So, you feed your network a sequence like:  
变形金刚通过一种称为位置编码的创新技术绕过了这一障碍。这个想法是获取输入序列中的所有单词——在这种情况下是一个英语句子——并为每个单词附加一个它的顺序。因此，您为网络提供如下序列：

`[("Dale", 1), ("says", 2), ("hello", 3), ("world", 4)]`

Conceptually, you can think of this as moving the burden of understanding word order from the structure of the neural network to the data itself.  
从概念上讲，您可以将此视为将理解词序的负担从神经网络的结构转移到数据本身。

At first, before the Transformer has been trained on any data, it doesn’t know how to interpret these positional encodings. But as the model sees more and more examples of sentences and their encodings, it learns how to use them effectively.  
起初，在 Transformer 接受任何数据训练之前，它不知道如何解释这些位置编码。但随着模型看到越来越多的句子示例及其编码，它会学习如何有效地使用它们。

I’ve done a bit of over-simplification here–the original authors used sine functions to come up with positional encodings, not the simple integers 1, 2, 3, 4–but the point is the same. Store word order as data, not structure, and your neural network becomes easier to train.  
我在这里做了一些过度简化——原作者使用正弦函数来进行位置编码，而不是简单的整数 1、2、3、4——但重点是一样的。将词序存储为数据而非结构，您的神经网络将变得更易于训练。

#### Attention 注意力

THE NEXT IMPORTANT PART OF TRANSFORMERS IS CALLED ATTENTION.  
变形金刚的下一个重要部分叫做注意力。

Got that? 了解？

Attention is a neural network structure that you’ll hear about all over the place in machine learning these days. In fact, the title of the 2017 paper that introduced Transformers wasn’t called, _We Present You the Transformer._ Instead it was called [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).  
注意力是一种神经网络结构，如今在机器学习中随处可见。事实上，2017 年介绍变形金刚的论文的标题并不是 “We Present You the Transformer”。相反，它被称为 Attention is All You Need 。

[Attention](https://arxiv.org/pdf/1409.0473.pdf) was introduced in the context of translation two years earlier, in 2015. To understand it, take this example sentence from the original paper:  
早在两年前，即 2015 年，在翻译的背景下引入了注意力。要理解它，请从原始论文中取出这个例句：

_The agreement on the European Economic Area was signed in August 1992.  
欧洲经济区协定于 1992 年 8 月签署。_

Now imagine trying to translate that sentence into its French equivalent:  
现在想象一下，试图将这句话翻译成对应的法语：

_L’accord sur la zone économique européenne a été signé en août 1992.  
欧洲经济区协定于 1992 年 8 月签署。_

One bad way to try to translate that sentence would be to go through each word in the English sentence and try to spit out its French equivalent, one word at a time.  
尝试翻译该句子的一种糟糕方法是遍历英语句子中的每个单词，然后尝试吐出其对应的法语，一次一个单词。  
That wouldn’t work well for several reasons, but for one, some words in the French translation are flipped: it’s “European Economic Area” in English, but “la zone économique européenne” in French. Also, French is a language with gendered words.  
出于多种原因，这种做法效果不佳，但其中一个原因是，法语翻译中的某些词被颠倒了：英语是 “欧洲经济区”，但法语是 “la zone économique européenne”。此外，法语是一种带有性别词的语言。  
The adjectives “économique” and “européenne” must be in feminine form to match the feminine object “la zone.”  
形容词 “économique” 和“européenne”必须是阴性形式才能匹配阴性宾语“la zone”。

Attention is a mechanism that allows a text model to “look at” every single word in the original sentence when making a decision about how to translate words in the output sentence. Here’s a nice visualization from that original attention paper:  
注意力是一种机制，它允许文本模型在决定如何翻译输出句子中的单词时 “查看” 原始句子中的每个单词。这是原始注意力论文的一个很好的可视化：

![](https://daleonai.com/images/screen-shot-2021-05-06-at-12.40.39-pm.png)

_Figure from the paper, “Neural Machine Translation by Jointly Learning to Align and Translate (2015)”  
图片来自论文 “通过联合学习对齐和翻译的神经机器翻译 (2015)”_

It’s a sort of heat map that shows where the model is “attending” when it outputs each word in the French sentence. As you might expect, w**h**en the model outputs the word “européenne,” it’s attending heavily to both the input words “European” and “Economic.”  
这是一种热图，显示模型在输出法语句子中的每个单词时 “参与” 的位置。正如您所料，当模型输出单词 “européenne” 时，它会同时关注输入单词 “European” 和“Economic”。

And how does the model know which words it should be “attending” to at each time step? It’s something that’s learned from training data. By seeing thousands of examples of French and English sentences, the model learns what types of words are interdependent.  
模型如何知道在每个时间步它应该 “关注” 哪些词？这是从训练数据中学到的东西。通过查看数千个法语和英语句子示例，该模型了解哪些类型的单词是相互依赖的。  
It learns how to respect gender, plurality, and other rules of grammar.  
它学习如何尊重性别、多元化和其他语法规则。

The attention mechanism has been an extremely useful tool for natural language processing since its discovery in 2015, but in its original form, it was used alongside recurrent neural networks. So, the innovation of the 2017 Transformers paper was, in part, to ditch RNNs entirely.  
自 2015 年发现以来，注意力机制一直是自然语言处理中非常有用的工具，但在其原始形式中，它与循环神经网络一起使用。因此，2017 年变形金刚论文的创新部分是完全抛弃 RNN。  
That’s why the 2017 paper was called “Attention is _all_ you need.”  
这就是为什么 2017 年的论文被称为 “Attention is all you need”。

#### Self-Attention 自注意力

The last (and maybe most impactful) piece of the Transformer is a twist on attention called “self-attention.”  
Transformer 的最后（也许也是最有影响力的）部分是对注意力的一种扭曲，称为 “自注意力”。

The type of “vanilla” attention we just talked about helped align words across English and French sentences, which is important for translation.  
我们刚刚谈到的 “普通” 注意力类型有助于对齐英语和法语句子中的单词，这对翻译很重要。  
But what if you’re not trying to translate words but instead build a model that understands underlying meaning and patterns in language–a type of model that could be used to do any number of language tasks?  
但是，如果您不是在尝试翻译单词，而是构建一个理解语言中潜在含义和模式的模型——一种可用于执行任意数量的语言任务的模型，该怎么办？

In general, what makes neural networks powerful and exciting and cool is that they often automatically build up meaningful internal representations of the data they’re trained on.  
总的来说，神经网络之所以强大、令人兴奋和酷，是因为它们经常自动为它们训练的数据建立有意义的内部表示。  
When you inspect the layers of a vision neural network, for example, you’ll find sets of neurons that “recognize” edges, shapes, and even high-level structures like eyes and mouths.  
例如，当您检查视觉神经网络的层时，您会发现一组神经元可以 “识别” 边缘、形状，甚至是眼睛和嘴巴等高级结构。  
A model trained on text data might automatically learn parts of speech, rules of grammar, and whether words are synonymous.  
在文本数据上训练的模型可能会自动学习词性、语法规则以及单词是否同义。

The better the internal representation of language a neural network learns, the better it will be at any language task. And it turns out that attention can be a very effective way of doing just this, if it’s turned on the input text itself.  
神经网络学习的语言内部表征越好，它在任何语言任务中的表现就越好。事实证明，注意力可以是一种非常有效的方式来做到这一点，如果它打开了输入文本本身的话。

For example, take these two sentence:  
例如，拿这两个句子：

“Server, can I have the check?”  
“服务员，可以给我支票吗？”

“Looks like I just crashed the server.”  
“看来我刚刚让服务器崩溃了。”

The word server here means two very different things, which we humans can easily disambiguate by looking at surrounding words. Self-attention allows a neural network to understand a word in the context of the words around it.  
这里的服务器一词意味着两个截然不同的事物，我们人类可以通过查看周围的词轻松地消除歧义。 Self-attention 允许神经网络在其周围单词的上下文中理解单词。

So when a model processes the word “server” in the first sentence, it might be “attending” to the word “check,” which helps disambiguate a human server from a metal one.  
因此，当模型处理第一句话中的 “服务器” 一词时，它可能会 “注意”“检查” 一词，这有助于消除人工服务器与金属服务器的歧义。

In the second sentence, the model might attend to the word “crashed” to determine _this_ “server” refers to a machine.  
在第二句话中，模型可能会注意 “崩溃” 一词，以确定这个 “服务器” 指的是一台机器。

Self-attention help neural networks disambiguate words, do part-of-speech tagging, entity resolution, learn semantic roles and [a lot more](https://arxiv.org/abs/1905.05950).  
自注意力帮助神经网络消除单词歧义、进行词性标记、实体解析、学习语义角色等等。

So, here we are.: Transformers, explained at 10,000 feet, boil down to:  
所以，我们在这里。：变形金刚，在 10,000 英尺的解释，归结为：

1.  Position Encodings 位置编码
2.  Attention 注意力
3.  Self-Attention 自注意力

If you want a deeper technical explanation, I’d highly recommend checking out Jay Alammar’s blog post [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/).  
如果您想要更深入的技术解释，我强烈建议您查看 Jay Alammar 的博文 The Illustrated Transformer。

#### What Can Transformers Do? 变形金刚能做什么？

One of the most popular Transformer-based models is called BERT, short for “Bidirectional Encoder Representations from Transformers.  
最受欢迎的基于 Transformer 的模型之一称为 BERT，是 “来自 Transformers 的双向编码器表示” 的缩写。  
” It was introduced by researchers at Google around the time I joined the company, in 2018, and soon made its way into almost every NLP project–including [Google Search](https://blog.google/products/search/search-language-understanding-bert/).  
” 它是在我于 2018 年加入公司时由谷歌的研究人员引入的，并很快进入几乎所有 NLP 项目——包括谷歌搜索。

BERT refers not just a model architecture but to a trained model itself, which you can download and use for free [here](https://github.com/google-research/bert). It was trained by Google researchers on a massive text corpus and has become something of a general-purpose pocket knife for NLP. It can be extended solve a bunch of different tasks, like:  
BERT 不仅指模型架构，还指训练模型本身，您可以在此处免费下载和使用。它由谷歌研究人员在大量文本语料库上进行训练，并已成为 NLP 的通用小刀。它可以扩展解决一堆不同的任务，比如：

\- text summarization- 文本摘要

\- question answering- 问答

\- classification- 分类

\- named entity resolution- 命名实体解析

\- text similarity- 文本相似度

\- offensive message/profanity detection  
- 冒犯性信息 / 亵渎检测

\- understanding user queries- 了解用户查询

\- a whole lot more- 更多

BERT proved that you could build very good language models trained on unlabeled data, like text scraped from Wikipedia and Reddit, and that these large “base” models could then be adapted with domain-specific data to lots of different use cases.  
BERT 证明你可以构建非常好的语言模型，在未标记的数据上进行训练，比如从维基百科和 Reddit 上抓取的文本，然后这些大型 “基础” 模型可以使用特定领域的数据来适应许多不同的用例。

More recently, the model [GPT-3](https://daleonai.com/gpt3-explained-fast), created by OpenAI, has been blowing people’s minds with its ability to generate realistic text. [Meena](https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html), introduced by Google Research last year, is a Transformer-based chatbot (akhem, “conversational agent”) that can have compelling conversations about almost any topic (this author once spent twenty minutes arguing with Meena about what it means to be human).  
最近，由 OpenAI 创建的模型 GPT-3 以其生成逼真的文本的能力而备受瞩目。谷歌研究院去年推出的 Meena 是一个基于 Transformer 的聊天机器人（akhem，“会话代理”），它可以就几乎任何话题进行引人入胜的对话（这位作者曾经花了 20 分钟与 Meena 争论人类的意义） .

Transformers have also been making waves outside of NLP, by [composing music](https://magenta.tensorflow.org/music-transformer), [generating images from text descriptions](https://daleonai.com/dalle-5-mins), and [predicting protein structure](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology).  
变形金刚也在 NLP 之外掀起波澜，通过创作音乐、从文本描述生成图像和预测蛋白质结构。

## How Can I Use Transformers? 如何使用变形金刚？

Now that you’re sold on the power of Transformers, you might want to know how you can start using them in your own app. No problemo.  
现在您已经了解了 Transformers 的强大功能，您可能想知道如何开始在您自己的应用程序中使用它们。没问题。

You can download common Transformer-based models like BERT from [TensorFlow Hub](https://tfhub.dev/). For a code tutorial, check out [this one](https://daleonai.com/semantic-ml) I wrote on building apps powered by semantic language.  
您可以从 TensorFlow Hub 下载常见的基于 Transformer 的模型，例如 BERT。有关代码教程，请查看我写的关于构建由语义语言支持的应用程序的教程。

But if you want to be really trendy and you write Python, I’d highly recommend the popular “Transformers” library maintained by the company [HuggingFace](https://huggingface.co/). The platform allow you to train and use most of today’s popular NLP models, like BERT, Roberta, T5, GPT-2, in a very developer-friendly way.  
但是，如果你想真正赶上潮流并编写 Python，我强烈推荐由 HuggingFace 公司维护的流行的 “变形金刚” 库。该平台允许您以对开发人员非常友好的方式训练和使用当今大多数流行的 NLP 模型，例如 BERT、Roberta、T5、GPT-2。

If you want to learn more about building apps with Transformers, come back soon! More tutorials coming soon.  
如果您想了解有关使用 Transformers 构建应用程序的更多信息，请尽快回来！更多教程即将推出。

* * *

Special thanks to Luiz/Gus Gustavo, Karl Weinmeister, and Alex Ku for reviewing early drafts of this post!  
特别感谢 Luiz/Gus Gustavo、Karl Weinmeister 和 Alex Ku 审阅本文的早期草稿！
