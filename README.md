# DRPDF

A Cloned Repository of PDFMathTranslate made to support RTL languages. DRPDF is a PDF scientific paper translation and bilingual comparison.

<img src="./docs/images/banner.png" width="320px"  alt="DRPDF"/>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0-orange)](LICENSE)
<a href="https://pypi.org/project/drpdf/">
<img src="https://img.shields.io/pypi/v/drpdf"></a>
<a href="https://pepy.tech/projects/drpdf">
<img src="https://static.pepy.tech/badge/drpdf"></a>
<a href="https://hub.docker.com/repository/docker/byaidu/drpdf">
<img src="https://img.shields.io/docker/pulls/byaidu/drpdf"></a>

<a href="https://trendshift.io/repositories/12424" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12424" alt="AI-Ahmed%2FDRPDF | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## Key Features

- 📊 **Layout Preservation**: Maintains formulas, charts, tables, and annotations in their original format
- 🌐 **Multi-Language Support**: Handles RTL languages and various translation services
- 🤖 **Multiple Interfaces**: CLI tool, GUI, Docker, and API access
- 🔄 **Smart Caching**: Efficient translation caching to speed up repeated operations
- 📑 **Document Analysis**: Advanced layout parsing with DocLayout-YOLO
- 🎯 **Customization**: Configurable translation services, prompts, and fonts

## Quick Start

### Installation

```bash
# Using UV (Recommended)
pip install uv
uv tool install --python 3.12 drpdf

# Using pip
pip install drpdf
```

### Basic Usage

```bash
# Translate a single PDF
drpdf document.pdf

# Use specific language pair
drpdf document.pdf -li en -lo ar

# Use GUI interface
drpdf -i

# Batch translate directory
drpdf --dir /path/to/pdfs/
```

### Docker Usage

```bash
# Pull and run
docker pull byaidu/drpdf
docker run -d -p 7860:7860 byaidu/drpdf

# Access web interface
open http://localhost:7860
```

## Advanced Features

### Translation Services

DRPDF supports multiple translation services:

```bash
# Using OpenAI
drpdf document.pdf -s openai

# Using DeepL
drpdf document.pdf -s deepl

# Using Google Translate
drpdf document.pdf -s google
```

### Language Support

```bash
# English to Arabic (RTL)
drpdf document.pdf -li en -lo ar

# English to Chinese
drpdf document.pdf -li en -lo zh

# Custom language pairs
drpdf document.pdf -li [source] -lo [target]
```

### Layout Options

```bash
# Skip font subsetting
drpdf document.pdf --skip-subset-fonts

# Use compatibility mode
drpdf document.pdf -cp

# Custom ONNX model
drpdf document.pdf --onnx path/to/model
```

## Python API

```python
from drpdf import translate

# Basic translation
translate(
    files=["document.pdf"],
    lang_in="en",
    lang_out="zh",
    service="google"
)

# Advanced options
translate(
    files=["document.pdf"],
    output="output_dir",
    pages=[1,2,3],  # Specific pages
    lang_in="en",
    lang_out="ar",
    service="openai",
    thread=4,  # Multi-threading
    compatible=True,  # Compatibility mode
    skip_subset_fonts=True,
    ignore_cache=False
)
```

## Configuration

Create a `config.json` for persistent settings:

```json
{
    "lang_in": "en",
    "lang_out": "zh",
    "service": "google",
    "thread": 4,
    "compatible": true,
    "skip_subset_fonts": false
}
```

Use with:
```bash
drpdf --config path/to/config.json document.pdf
```

- 📊 Preserve formulas, charts, table of contents, and annotations _([preview](#preview))_.
- 🌐 Support [multiple languages](#language) including RTL languages, and diverse [translation services](#services).
- 🤖 Provides [commandline tool](#usage), [interactive user interface](#gui), and [Docker](#docker)

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

<h2 id="updates">Updates</h2>

- [May 9, 2025] pdf2zh 2.0 Preview Version [#586](https://github.com/Byaidu/PDFMathTranslate/issues/586): The Windows ZIP file and Docker image are now available.

> [!CAUTION]
>
> The current maintainer of this project is developing pdf2zh v2.0-rc at [aw/pdf2zh v2-rc](https://github.com/awwaawwa/PDFMathTranslate/tree/v2-rc).
>
> v2.0-rc can now accept some small PRs. Please discuss with the maintainer [@awwaawwa](https://github.com/awwaawwa) in the [main repository issue](https://github.com/Byaidu/PDFMathTranslate/issues) before submitting a PR to [aw/pdf2zh v2-rc](https://github.com/awwaawwa/PDFMathTranslate/tree/v2-rc)~

- [Mar. 3, 2025] Experimental support for the new backend [BabelDOC](https://github.com/funstory-ai/BabelDOC) WebUI added as an experimental option (by [@awwaawwa](https://github.com/awwaawwa))
- [Feb. 22 2025] Better release CI and well-packaged windows-amd64 exe (by [@awwaawwa](https://github.com/awwaawwa))
- [Dec. 24 2024] The translator now supports local models on [Xinference](https://github.com/xorbitsai/inference) _(by [@imClumsyPanda](https://github.com/imClumsyPanda))_
- [Dec. 19 2024] Non-PDF/A documents are now supported using `-cp` _(by [@reycn](https://github.com/reycn))_
- [Dec. 13 2024] Additional support for backend by _(by [@YadominJinta](https://github.com/YadominJinta))_
- [Dec. 10 2024] The translator now supports OpenAI models on Azure _(by [@yidasanqian](https://github.com/yidasanqian))_

<h2 id="preview">Preview</h2>

<div align="center">
<img src="./docs/images/preview.gif" width="80%"/>
</div>

<h2 id="demo">Online Service 🌟</h2>

You can try our application out using either of the following demos:

- [Public free service](https://drpdf.com/) online without installation _(recommended)_.
- [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) 1000 free pages per month. _(recommended)_
- [Demo hosted on HuggingFace](https://huggingface.co/spaces/reycn/DRPDF-Docker)
- [Demo hosted on ModelScope](https://www.modelscope.cn/studios/AI-ModelScope/DRPDF) without installation.

Note that the computing resources of the demo are limited, so please avoid abusing them.

<h2 id="install">Installation and Usage</h2>

### Methods

For different use cases, we provide distinct methods to use our program:

<details open>
  <summary>1. UV install</summary>

1. Python installed (3.10 <= version <= 3.12)

2. Install our package:

   ```bash
   pip install uv
   uv tool install --python 3.12 drpdf
   ```

3. Execute translation, files generated in [current working directory](https://chatgpt.com/share/6745ed36-9acc-800e-8a90-59204bd13444):

   ```bash
   drpdf document.pdf
   ```

</details>

<details>
  <summary>2. Windows exe</summary>

1. Download drpdf-version-win64.zip from [release page](https://github.com/AI-Ahmed/DRPDF/releases)

2. Unzip and double-click `drpdf.exe` to run.

</details>

<details>
  <summary>3. Graphic user interface</summary>

1. Python installed (3.10 <= version <= 3.12)

2. Install our package:

```bash
pip install drpdf
```

3. Start using in browser:

   ```bash
   drpdf -i
   ```

4. If your browswer has not been started automatically, goto

   ```bash
   http://localhost:7860/
   ```

   <img src="./docs/images/gui.gif" width="500"/>

See [documentation for GUI](./docs/README_GUI.md) for more details.

</details>

<details>
  <summary>4. Docker</summary>

1. Pull and run:

   ```bash
   docker pull byaidu/drpdf
   docker run -d -p 7860:7860 byaidu/drpdf
   ```

2. Open in browser:

   ```
   http://localhost:7860/
   ```

For docker deployment on cloud service:

<div>
<a href="https://www.heroku.com/deploy?template=https://github.com/Byaidu/PDFMathTranslate">
  <img src="https://www.herokucdn.com/deploy/button.svg" alt="Deploy" height="26"></a>
<a href="https://render.com/deploy">
  <img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Koyeb" height="26"></a>
<a href="https://zeabur.com/templates/5FQIGX?referralCode=reycn">
  <img src="https://zeabur.com/button.svg" alt="Deploy on Zeabur" height="26"></a>
<a href="https://template.sealos.io/deploy?templateName=pdf2zh">
  <img src="https://sealos.io/Deploy-on-Sealos.svg" alt="Deploy on Sealos" height="26"></a>
<a href="https://app.koyeb.com/deploy?type=git&builder=buildpack&repository=github.com/Byaidu/PDFMathTranslate&branch=main&name=pdf-math-translate">
  <img src="https://www.koyeb.com/static/images/deploy/button.svg" alt="Deploy to Koyeb" height="26"></a>
</div>

</details>

<details>
  <summary>5. Zotero Plugin</summary>


See [Zotero PDF2zh](https://github.com/guaguastandup/zotero-pdf2zh) for more details.

</details>

<details>
  <summary>6. Commandline</summary>

1. Python installed (3.10 <= version <= 3.12)
2. Install our package:

   ```bash
   pip install drpdf
   ```

3. Execute translation, files generated in [current working directory](https://chatgpt.com/share/6745ed36-9acc-800e-8a90-59204bd13444):

   ```bash
   drpdf document.pdf
   ```

</details>

> [!TIP]
>
> - If you're using Windows and cannot open the file after downloading, please install [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) and try again.
>
> - If you cannot access Docker Hub, please try the image on [GitHub Container Registry](https://github.com/AI-Ahmed/DRPDF/pkgs/container/drpdf).
> ```bash
> docker pull ghcr.io/ai-ahmed/drpdf
> docker run -d -p 7860:7860 ghcr.io/ai-ahmed/drpdf
> ```

### Unable to install?

The present program needs an AI model(`wybxc/DocLayout-YOLO-DocStructBench-onnx`) before working and some users are not able to download due to network issues. If you have a problem with downloading this model, we provide a workaround using the following environment variable:

```shell
set HF_ENDPOINT=https://hf-mirror.com
```

For PowerShell user:

```shell
$env:HF_ENDPOINT = https://hf-mirror.com
```

If the solution does not work to you / you encountered other issues, please refer to [frequently asked questions](https://github.com/Byaidu/PDFMathTranslate/wiki#-faq--%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98).

<h2 id="usage">Advanced Options</h2>

Execute the translation command in the command line to generate the translated document `example-mono.pdf` and the bilingual document `example-dual.pdf` in the current working directory. Use Google as the default translation service. More support translation services can find [HERE](https://github.com/AI-Ahmed/DRPDF/blob/main/docs/ADVANCED.md#services).

<img src="./docs/images/cmd.explained.png" width="580px"  alt="cmd"/>

In the following table, we list all advanced options for reference:

| Option | Function | Example |
|--------|----------|---------|
| `files` | Local files | `drpdf ~/local.pdf` |
| `links` | Online files | `drpdf http://arxiv.org/paper.pdf` |
| `-i` | [Enter GUI](#gui) | `drpdf -i` |
| `-p` | [Partial document translation](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#partial) | `drpdf example.pdf -p 1` |
| `-li` | [Source language](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#languages) | `drpdf example.pdf -li en` |
| `-lo` | [Target language](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#languages) | `drpdf example.pdf -lo zh` |
| `-s` | [Translation service](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#services) | `drpdf example.pdf -s deepl` |
| `-t` | [Multi-threads](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#threads) | `drpdf example.pdf -t 1` |
| `-o` | Output dir | `drpdf example.pdf -o output` |
| `-f`, `-c` | [Exceptions](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#exceptions) | `drpdf example.pdf -f "(MS.*)"` |
| `-cp` | Compatibility Mode | `drpdf example.pdf --compatible` |
| `--skip-subset-fonts` | [Skip font subset](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#font-subset) | `drpdf example.pdf --skip-subset-fonts` |
| `--ignore-cache` | [Ignore translate cache](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cache) | `drpdf example.pdf --ignore-cache` |
| `--share` | Public link | `drpdf -i --share` |
| `--authorized` | [Authorization](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#auth) | `drpdf -i --authorized users.txt [auth.html]` |
| `--prompt` | [Custom Prompt](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#prompt) | `drpdf --prompt [prompt.txt]` |
| `--onnx` | Use Custom DocLayout-YOLO ONNX model | `drpdf --onnx [onnx/model/path]` |
| `--serverport` | Use Custom WebUI port | `drpdf --serverport 7860` |
| `--dir` | Batch translate | `drpdf --dir /path/to/translate/` |
| `--config` | [Configuration file](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cofig) | `drpdf --config /path/to/config/config.json` |
| `--babeldoc` | Use Experimental backend [BabelDOC](https://funstory-ai.github.io/BabelDOC/) | `drpdf --babeldoc -s openai example.pdf` |
| `--mcp` | Enable MCP STDIO mode | `drpdf --mcp` |
| `--sse` | Enable MCP SSE mode | `drpdf --mcp --sse` |

For detailed explanations, please refer to our document about [Advanced Usage](./docs/ADVANCED.md) for a full list of each option.

<h2 id="downstream">Secondary Development (APIs)</h2>

For downstream applications, please refer to our document about [API Details](./docs/APIS.md) for futher information about:

- [Python API](./docs/APIS.md#api-python), how to use the program in other Python programs
- [HTTP API](./docs/APIS.md#api-http), how to communicate with a server with the program installed

<h2 id="todo">TODOs</h2>

- [ ] Parse layout with DocLayNet based models, [PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/17cc27ac3842e7880ca4aad92358d3ef8555429a/paddlex/repo_apis/PaddleDetection_api/object_det/official_categories.py#L81), [PaperMage](https://github.com/allenai/papermage/blob/9cd4bb48cbedab45d0f7a455711438f1632abebe/README.md?plain=1#L102), [SAM2](https://github.com/facebookresearch/sam2)

- [ ] Fix page rotation, table of contents, format of lists

- [ ] Fix pixel formula in old papers

- [ ] Async retry except KeyboardInterrupt

- [ ] Knuth–Plass algorithm for western languages

- [ ] Support non-PDF/A files

- [ ] Plugins of [Zotero](https://github.com/zotero/zotero) and [Obsidian](https://github.com/obsidianmd/obsidian-releases)

<h2 id="acknowledgement">Acknowledgements</h2>

- [Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors to this project, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

- New backend: [BabelDOC](https://github.com/funstory-ai/BabelDOC)

- Document merging: [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

- Document parsing: [Pdfminer.six](https://github.com/pdfminer/pdfminer.six)

- Document extraction: [MinerU](https://github.com/opendatalab/MinerU)

- Document Preview: [Gradio PDF](https://github.com/freddyaboulton/gradio-pdf)

- Multi-threaded translation: [MathTranslate](https://github.com/SUSYUSTC/MathTranslate)

- Layout parsing: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

- Document standard: [PDF Explained](https://zxyle.github.io/PDF-Explained/), [PDF Cheat Sheets](https://pdfa.org/resource/pdf-cheat-sheets/)

- Multilingual Font: [Go Noto Universal](https://github.com/satbyy/go-noto-universal)

<h2 id="contrib">Contributors</h2>

<a href="https://github.com/AI-Ahmed/DRPDF/graphs/contributors">
  <img src="https://opencollective.com/DRPDF/contributors.svg?width=890&button=false" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/dfa7583da5332a11468d686fbd29b92320a6a869.svg "Repobeats analytics image")

<h2 id="star_hist">Star History</h2>

<a href="https://star-history.com/#AI-Ahmed/DRPDF&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=AI-Ahmed/DRPDF&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=AI-Ahmed/DRPDF&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=AI-Ahmed/DRPDF&type=Date"/>
 </picture>
</a>
