from app.main import build_scan_response, map_category


def test_maps_specific_classes_to_three_main_categories():
    assert map_category("biological") == "Organik"
    assert map_category("paper") == "Organik"
    assert map_category("plastic") == "Anorganik"
    assert map_category("metal") == "Anorganik"
    assert map_category("battery") == "B3"
    assert map_category("trash") == "B3"


def test_response_keeps_frontend_compatibility():
    response = build_scan_response(
        filename="sample.jpg",
        predicted_label="plastic",
        confidence=0.91,
        raw_predictions=[0.09, 0.91],
    )

    assert response["predicted_class"] == "Anorganik"
    assert response["specific_class"] == "plastic"
    assert response["category"] == "Anorganik"
    assert response["confidence"] == 0.91
