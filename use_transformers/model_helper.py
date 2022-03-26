from transformers.models.bert import TFBertForSequenceClassification
from transformers.models.bert.modeling_tf_bert import *
from transformers.modeling_tf_utils import shape_list

import tensorflow as tf
import tensorflow.keras as keras


_CHECKPOINT_FOR_DOC = "bert-base-cased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    自定义计算输出的loss
    """

    def hf_compute_loss(self, labels, logits):
        """
        既然在这里计算了损失
        """
        if len(shape_list(logits)) == 1 or shape_list(logits)[1] == 1:
            loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
        else:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.NONE
            )

        return loss_fn(labels, logits)


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    """
    自定义模型
    """

    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name="bert")
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = keras.layers.Dropout(rate=classifier_dropout)
        self.dropout2 = keras.layers.Dropout(rate=classifier_dropout)
        self.classifier1 = keras.layers.Dense(
            units=6,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier1",
        )
        self.classifier2 = keras.layers.Dense(
            units=25,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier2",
        )

    # @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]

        # 第一个输出
        drop_output1 = self.dropout1(inputs=pooled_output, training=training)
        logits1 = self.classifier1(inputs=pooled_output)
        loss1 = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits1)

        # 第二个输出
        drop_output2 = self.dropout2(inputs=drop_output1, training=training)
        logits2 = self.classifier2(inputs=drop_output2)
        loss2 = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits2)

        # 返回有两种格式, 一种是按类字典类返回, 一种是按元组返回
        if not return_dict:
            output = (logits1, logits2) + outputs[2:]
            if loss1 and loss2:
                output = (loss1, loss2) + output
            # 这个真的只有两个输出, 后面的 output 是空的
            # TODO: 应该看看怎么把这个 output 的名字换了, 不然用 output_1 和 output_2 很难受的, 需要数据集里也使用这个名字
            return output

        return TFSequenceClassifierOutput(
            loss=(loss1, loss2),
            logits=(logits1, logits2),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)
