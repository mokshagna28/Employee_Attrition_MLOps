from src.train import train_model


def test_model_trains_successfully():
    model = train_model()
    assert model is not None


def test_model_predicts_attrition():
    model = train_model()
    prediction = model.predict([[1, 2]])
    assert prediction[0] in [0, 1]
