import tensorflow_model_analysis as tfma

# saved model을 평가 가능하도록 eval shared model로 변환
eval_shared_model = tfma.default_eval_shared_model(
      eval_svaed_model_path=_MODEL_DIR,
      tags=[tf.saved.model.SERVING]
)

# eval config 제공, label 정보 및 표시할 모든 지표 정의
eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='consumer_disputed')],
      slicing_specs[tfma.SlicingSpec()],
      metrics_specs=[
            tfma.MetricsSpec(metrics=[
                  tfma.MetricConfig(class_name='BinaryAccuracy'),
                  tfma.MetricConfig(class_name='ExampleCount'),
                  tfma.MetricConfig(class_name='FalsePositives'),
                  tfma.MetricConfig(class_name='TruePositives'),
                  tfma.MetricConfig(class_name='FalseNegatives'),
                  tfma.MetricConfig(class_name='TrueNegatives'),
            ])
      ]
)

# 모델 분석 단계 실행
eval_result = tfma.run_model_analysis(
      eval_shared_model=eval_shared_model,
      eval_config=eval_config,
      data_location=_EVAL_DATA_FILE,
      output_path=_EVAL_RESULT_LOCATION,
      file_format='tfrecords')

tfma.view.render_slicing_metrcis(eval_result)