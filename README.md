# Car Classification with EfficientNetB0

## Introduction
This is a mini-project of implementing Classification with Stanford Cars Dataset using Transfer learning method and EfficientNetB0 as the backbone.

## Tutorial

Clone the project

```bash
git clone https://github.com/zogojogo/car-classification-wii.git
```

Go to the project directory

```bash
cd car-classify-kris
```

Download Dependencies
```bash
pip install -r requirements.txt
```

Start API service

```
python3 app.py
```
  
## API Reference

Service: http://your-ip-address:8080

#### POST image

```http
  POST /predict_car
```
Content-Type: multipart/form-data
| Name    | Type   | Description                                         |
| :------ | :----- | :-------------------------------------------------- |
| `image` | `file` | **Required**. `image/png` MIME Type |

## Output Example

**Output:**<br>
```python
{
  "filename": "<filename>",
  "contentype": "<image type>",
  "predicted class": "<string>",
  "confidence": "<float>",
  "inference time": "<inference time>"
}
```