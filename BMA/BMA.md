摘要：
多模型集合策略是利用不同模型的技能预测的多样性的一种手段。本文研究并使用贝叶斯模型平均法（BMA）方法，从多个水文模型的对比预测中开发出更加熟练和可靠的概率性水文预测结果。
BMA是一种统计程序，它通过对单个预测的权衡，推断出一致的预测结果，衡量各个预测，表现较好的预测比表现较差的预测获得更高的权重。绩效好的预测比绩效差的预测获得更高的权重。此外，BMA对总的预测不确定性的描述比原始集合更可靠的描述，从而使概率密度函数（PDF）更清晰，校准得更好。在这项研究中，一个由9个成员组成的水文预测集合被用来测试和评估BMA方案。该组合是通过使用三个不同的目标函数校准三个不同的水文模型而产生的。这些目标函数的选择方式能使模型很好地捕捉水文图的某些方面（例如，峰值、中段流量和低流量）。
在美国的三个测试流域进行了两组数值实验，以探索使用BMA的最佳方式。
在第一组实验中，计算了一组BMA权重，以获得BMA预测结果，而第二组实验则采用了多组权重，不同的权重组有不同的预测。
在这两组中，流速值都采用Box-Cox转换法以确保预测误差的概率分布是近似高斯的。分割采样方法来获得和验证BMA预测结果。测试结果表明，BMA方案具有以下优势：与原始合集相比，BMA方案具有产生更熟练和同样可靠的概率预测的优势。预期的性能在日均方根误差（DRMS）和日绝对平均误差（DABS）方面，BMA预测的性能普遍优于最佳个体预测。胜过最好的单个预测结果。此外，采用多组权重的BMA预测通常比采用单组权重的BMA预测要好。

正文：
1.引言
到目前为止，水文学家的普遍做法是依靠一个单一的水文模型来进行水文预测。尽管在开发更多的水文模型方面投入了巨大的资源来开发更多的水文模型。没有人能够令人信服地声称，今天存在的任何模型在所有类型的应用和所有条件下都优于其他模型。所有类型的应用和在所有条件下都优于其他模型，不同的模型在捕捉水文过程的不同方面具有优势。依靠单一模型，往往会导致预测结果以牺牲某些现象或事件为代价，导致一些现象或事件被很好地表现出来，而其他现象或事件则被牺牲掉。此外，对与这些预测有关的不确定性还没有得到足够的重视。基于多参数集的集合方法可以帮助改善不确定性的估计。但在这种集合策略中，任何单一模型所固有的结构误差是无法避免的。这种集合策略。这就促使一些研究人员提倡多模型方法。

多模型方法被应用于各种预测的应用，如经济和天气预测
早在20世纪60年代。Shamseldin及其同事可能是第一个探索使用多模型方法进行水文预测的人。Georgakakos等人最近使用了一种多模型组合的方法来分析来自参与分布式模型的多个模型的模拟结果.这些多模型技术通过线性组合各个模型的预测结果，根据不同的加权策略。在最简单的情况下，所有模型的权重可以是相同的或者通过某些基于回归的方法。在后一种情况下，权重是回归系数。Shamseldin和
O'Connor也探索了使用人工神经网络（ANN）技术来估计模型的方法。网络（ANN）技术来估计模型权重。Raftery等人指出，那些基于回归技术的权重测定的权重很难解释
解释，因为它们具有任意的负值或正值，而且与模型性能没有关系。此外，这些方法的多模型预测的可靠性并不令人满意。然而，在根据各种预测技能和可靠性进行评估时，这些方法产生的多模型集合平均数一直比单一模型的预测结果要好。判断时，这些方法产生的集合平均数的表现一直优于单一模型的预测，因为它们基于各种预测技能和可靠性分数。

最近，贝叶斯模型平均法(BMA)在统计学、管理学等不同领域得到了普及。在不同的领域，如统计学、管理学、医学和气象学、医学和气象学。如同和其他多模型方法的预测一样，BMA预测是来自竞争模型的单个预测的加权平均数。但有一些BMA也提供了一个更真实的对预测的不确定性的描述，说明解释了模型间差异和模型内差异。BMA的权重都是正的，总和为1。体现了相对的模型性能，因为它们是一个模型的概率性，可能性，观察结果的正确性。在各种案例研究中。BMA已被证明能产生比其他多模型更准确、更可靠的
在各种案例研究中，BMA被证明比其他多模型技术产生更准确和可靠的预测结果；最近，BMA方法也被应用于水文应用中，如 Neuman 和 Wierenga 的地下水建模。

本研究探讨了BMA在水文方面的应用。我们感兴趣的是如何利用BMA方案来提高水流预测的准确性和流水预测的准确性和可靠性。特别是，我们研究了应用BMA方案的不同方法，以充分充分利用各个模型的优势。本文主要内容如下
组织如下. 第2节介绍了BMA方法。第3节讨论了水文模型组合的生成和数值试验的设计.模型的生成以及数值实验和测试数据集的设计和测试数据集。第4节描述了BMA方案的测试和验证结果。第5节提供了总结和结论。

2.BMA
贝叶斯模型平均法（BMA）是一种统计方案。推断出一个概率预测，该预测拥有比原始集合成员具备更多的技能和可靠性，而不是由几个相互竞争的模型产生的原始集合成员
竞争的模型。BMA已被主要用于广义的线性回归应用。最近，Raftery等人成功地将BMA在动态建模应用中（即，数值天气预测）。在这项研究中，我们将BMA应用于流水预测问题。简要描述BMA方案如下：