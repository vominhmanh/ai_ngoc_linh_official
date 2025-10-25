curl 'https://api.thucchien.ai/gemini/v1beta/models/gemini-2.5-flash-image-preview:generateContent' \
-H 'x-goog-api-key: 
<your_api_key>
' \
-H 'Content-Type: application/json' \
-d '{
  "contents": [{
    "parts": [
      {"text": "
A photorealistic close-up portrait of an elderly Japanese ceramicist with deep, sun-etched wrinkles and a warm, knowing smile. He is carefully inspecting a freshly glazed tea bowl. The setting is his rustic, sun-drenched workshop with pottery wheels and shelves of clay pots in the background. The scene is illuminated by soft, golden hour light streaming through a window, highlighting the fine texture of the clay and the fabric of his apron. Captured with an 85mm portrait lens, resulting in a soft, blurred background (bokeh). The overall mood is serene and masterful.
"}
    ]
  }],
  "generationConfig": {
      "imageConfig": {
          "aspectRatio": "
9:16
"
      }
  }
}'