# Third-party notices

video-note contains code ported from other projects of the same author. Each ported file also
carries this notice at its top. The ported copies are pinned to the commits below; they do not
follow later changes of the originals.

## sipher

- Source: sipher (repository of the same author), version 0.1.7, commit `dc21664389696b85daf440932d9abbbcd5c111cb`
- License: MIT — Copyright (c) 2026 stepbyjason-lab

| Ported file here | Original | Changes |
|---|---|---|
| `scripts/ocr_gemini.py` | `core/llm_free.py` | HTTP through urllib instead of `requests`; settings from the video-note key file; language helper replaced by `VIDEO_NOTE_LANG`; model always passed by the caller |
| `scripts/ocr_ensemble.py` | `core/ocr_ensemble.py` | same HTTP and settings changes; built-in default rosters; paid-Claude fallback, console consent prompt and `.env.local` writes removed; batch helper `ocr_images()` added |

### License text (sipher, MIT)

```
MIT License

Copyright (c) 2026 stepbyjason-lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
