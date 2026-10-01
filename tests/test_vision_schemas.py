from backend.schemas.vision import VisionDimension, VisionPageResult


def test_vision_page_schema_accepts_semantic_dimension():
    page = VisionPageResult.model_validate({
        "page_number": 1,
        "units": "m",
        "regions": [
            {"id": "site_1", "type": "SITE_PLAN", "bbox": [1, 2, 3, 4], "confidence": 0.95}
        ],
        "dimensions": [
            {
                "value": 9.14,
                "unit": "m",
                "type": "PLOT_WIDTH",
                "region_id": "site_1",
                "bbox": [10, 20, 30, 40],
                "evidence": "9.14",
                "confidence": 0.96,
            }
        ],
    })
    assert page.dimensions[0].value == 9.14
    assert page.dimensions[0].type == "PLOT_WIDTH"
