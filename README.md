# VOICEVOX on RX 9070 XT (gfx1201) — ORT 1.17.3 fake-CUDA ROCm EP patches

VOICEVOX engine 0.25.x (linux-x64-nvidia 配布版) を AMD Radeon RX 9070 XT
(Navi 48 / RDNA 4 / gfx1201) 上で **CUDA EP のフリをした ROCm EP** に差し替えて
動かすための、ONNX Runtime v1.17.3 (VOICEVOX/onnxruntime フォークと同等) への
最小パッチセットです。

実機検証で、合成出力は CPU リファレンスとの SNR +81〜83 dB を達成 (実質一致)、
長文 synthesis 818ms / 18× リアルタイム、RTX 3060 比 2.03× の時間で動作することを
確認しています。

本 README はパッチを適用してビルド・デプロイするための実用手順だけを書きます。

---

## 含まれるファイル

| ファイル | 内容 |
|---|---|
| `01-essential.patch` | 本質的な機能変更 (8 ファイル) |
| `02-build.patch` | ビルド系の互換性パッチ (6 ファイル) |
| `README.md` | 本文書 |

### 01-essential.patch (本質変更)

| 対象ファイル | 変更内容 |
|---|---|
| `include/onnxruntime/core/graph/constants.h` | `kRocmExecutionProvider = "CUDAExecutionProvider"` に変更 (fake-CUDA disguise の核) |
| `onnxruntime/core/providers/shared_library/provider_api.h` | 同上 (provider plugin が実際に参照する第二の定義) |
| `onnxruntime/core/providers/rocm/cu_inc/common.cuh` | **`GPU_WARP_SIZE = 32`** (RDNA wave32 / 最重要・正しい出力の必要条件) |
| `onnxruntime/core/providers/rocm/nn/conv.cc` | MIOpen `GetWorkSpaceSize` が gfx1201 で 0 を返す問題のフォールバック (512 MiB cap) + `exhaustive_search` を provider option で制御可能に |
| `onnxruntime/core/providers/rocm/nn/conv_transpose.cc` | 同 workspace fallback と exhaustive_search 反映 |
| `onnxruntime/core/providers/rocm/rocm_execution_provider.cc` | 診断用環境変数 (`ROCM_DENY_OPS` / `ROCM_ALLOW_OPS_ONLY` / `ROCM_LOG_OP_PLACEMENT`) を `GetCapability()` に追加 |
| `onnxruntime/core/providers/rocm/rocm_execution_provider_info.cc` | CUDA 用 provider option (`cudnn_conv_algo_search`, `enable_cuda_graph`, `user_compute_stream` 等) を no-op で受理、`miopen_conv_exhaustive_search` に alias |
| `onnxruntime/core/providers/rocm/rocm_execution_provider_info.h` | `miopen_conv_exhaustive_search` 既定値を true に。`TunableOpInfo` に RTTI 制約のコメントを追記 |

### 02-build.patch (ビルド対応)

| 対象ファイル | 変更内容 |
|---|---|
| `cmake/CMakeLists.txt` | ROCm 7.x の `lib64/cmake/*` パス追加、`.info/version-dev` ファイル欠落時のフォールバック、`CMP0169` (cmake 4.x 対応) |
| `cmake/deps.txt` | eigen ZIP の SHA1 を gitlab 側の現在値に追従 |
| `cmake/external/onnxruntime_external_deps.cmake` | システムインストール版を拾わせない (`FIND_PACKAGE_ARGS` を除去して bundled 版に固定) |
| `cmake/onnxruntime_providers_rocm.cmake` | RCCL / roctracer を optional に (Gentoo の ROCm 7.2 では一部が分離パッケージ) |
| `onnxruntime/core/providers/cuda/tensor/gather_elements.cc` | hipify 後の unused variable 警告を warning-as-error から救う `ORT_UNUSED_PARAMETER` |
| `onnxruntime/core/providers/rocm/integer_gemm.cc` | `dynamic_cast<RocmStream*>` → `static_cast` (`-fno-rtti` 下でのビルドエラー回避) |

---

## 前提条件

- AMD GPU で gfx1201 (RX 9070 XT 等 RDNA 4) を使う想定
  - 別アーキで使う場合は `01-essential.patch` 内の `GPU_WARP_SIZE` 値を変更
    (CDNA wave64 = 64, RDNA wave32 = 32, 他 NVIDIA 互換 = 32)
- ROCm 7.x (検証は 7.2.53210)
  - `dev-util/hip`, `sci-libs/rocBLAS`, `sci-libs/MIOpen`, `sci-libs/hipFFT`,
    `sci-libs/rocRAND` がインストール済み
  - 推奨: `sci-libs/MIOpen` を `composable-kernel` USE flag 有効でビルド
    (Gentoo ebuild の REQUIRED_USE をパスするため CDNA target も併用が必要)
- CMake 3.26 以上 (検証は cmake 4.x でも動作確認済み、CMP0169 で対応)
- Clang 22 (ROCm 7.2 同梱の AMD LLVM)
- `hipify-perl` — `dev-util/hipify-clang` (Portage) でインストールされる
  `/usr/bin/hipify-perl` を使用。未インストールの場合は管理者に emerge を依頼するか、
  GitHub Releases (`ROCm/HIPIFY`) からスクリプト単体をダウンロードして
  `-Donnxruntime_HIPIFY_PERL=/path/to/hipify-perl` で cmake に渡す
- VOICEVOX engine 0.25.x の linux-x64-nvidia 配布版、または `VOICEVOX.AppImage` が手元にあること

---

## ビルド手順

### 1. ソース取得とパッチ適用

```bash
# ベースは v1.17.3 (microsoft/onnxruntime と VOICEVOX/onnxruntime のいずれでも可)
git clone --depth 1 --branch v1.17.3 https://github.com/microsoft/onnxruntime.git ort-fake-cuda
cd ort-fake-cuda

# パッチ適用 (順序は問わない)
git apply /path/to/voicevox-rocm-patches/01-essential.patch
git apply /path/to/voicevox-rocm-patches/02-build.patch

# サブモジュール
git submodule update --init --recursive
```

### 2. CMake 設定

ROCm 7.2 / Gentoo / gfx1201 を想定:

```bash
mkdir -p build/Release && cd build/Release

cmake ../../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_USE_ROCM=ON \
  -Donnxruntime_ROCM_HOME=/usr \
  -Donnxruntime_ROCM_VERSION=7.2 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_HIP_COMPILER=/usr/lib/llvm/22/bin/clang++ \
  -Donnxruntime_BUILD_SHARED_LIB=ON \
  -Donnxruntime_BUILD_UNIT_TESTS=OFF \
  -Donnxruntime_DISABLE_RTTI=ON \
  -Donnxruntime_DISABLE_EXCEPTIONS=OFF \
  -Donnxruntime_ENABLE_PYTHON=OFF \
  -Donnxruntime_USE_COMPOSABLE_KERNEL=OFF \
  -Donnxruntime_DISABLE_CONTRIB_OPS=ON \
  -Donnxruntime_HIPIFY_PERL=/usr/bin/hipify-perl \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

- ROCm のインストール先が `/usr` 以外なら `-Donnxruntime_ROCM_HOME=` を調整
- 別アーキ向けには `-DCMAKE_HIP_ARCHITECTURES=` を変更
- `hipify-perl` を別の場所に置いた場合は `-Donnxruntime_HIPIFY_PERL=` を調整
- `-Donnxruntime_USE_COMPOSABLE_KERNEL=OFF`: composable_kernel は VOICEVOX の
  推論に不要かつ cmake 4.x との互換性問題があるため無効化
- `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`: cmake 4.x で依存ライブラリ
  (google_nsync 等) の古い cmake_minimum_required を許容するために必要

### 3. ビルド (provider プラグインのみ)

VOICEVOX で必要なのは provider plugin の `.so` だけです。
main lib (`libonnxruntime.so`) は VOICEVOX 同梱のものを使うので、ビルド対象を絞る:

```bash
cmake --build . --target onnxruntime_providers_rocm -j$(nproc)
```

成果物: `build/Release/libonnxruntime_providers_rocm.so` (約 50 MB)

ビルド時間: 並列度 16 で初回約 15 分、`.so` 単体の差分ビルドなら 1〜2 分。

---

## AppImage リパッケージ手順 (VOICEVOX.AppImage を ROCm 対応版に差し替える)

`~/.voicevox/VOICEVOX.AppImage` はコア・エンジン・エディタを 1 ファイルに統合した配布形式です。
内部の `vv-engine/libvoicevox_onnxruntime_providers_cuda.so` を ROCm ビルドに差し替えて
再パッケージすることで、AppImage のまま ROCm を利用できます。

### 必要ツール

- `squashfs-tools` (`unsquashfs` / `mksquashfs`)

### スクリプトで一発実行

```bash
./repackage-appimage.sh \
  --rocm-so /path/to/ort-fake-cuda/build/Release/libonnxruntime_providers_rocm.so \
  --appimage ~/.voicevox/VOICEVOX.AppImage \
  --output   ~/VOICEVOX-ROCm.AppImage
```

所要時間: squashfs 展開 + 再圧縮合わせて約 3〜5 分 (24 コア時)。
完了後は `~/VOICEVOX-ROCm.AppImage` を通常の AppImage と同様に実行するだけです。

```bash
HIP_VISIBLE_DEVICES=0 ~/VOICEVOX-ROCm.AppImage
```

> **ROCm ライブラリ解決について**: AppRunOriginal は `LD_LIBRARY_PATH` に `${APPDIR}/usr/lib` を
> 先頭追加するだけで、system ライブラリパスは保持されます。ROCm ライブラリ
> (`libamdhip64.so.7`, `libMIOpen.so.1` 等) が `/usr/lib64/` にインストール済みであれば
> 追加設定なしに解決されます。

### スクリプトの内部処理

| ステップ | 処理 |
|---|---|
| 1 | `--appimage-offset` でランタイムサイズ (944632 bytes) を取得 |
| 2 | `unsquashfs -o <offset>` で squashfs を展開 |
| 3 | `vv-engine/libvoicevox_onnxruntime_providers_cuda.so` を ROCm ビルド版で上書き |
| 3b | `vv-engine/engine_internal/libstdc++.so.6` をシステム版で置換 (後述) |
| 3c | `AppRun` に `export MIOPEN_FIND_MODE=FAST` を追記 (後述) |
| 4 | `mksquashfs -comp zstd -b 131072 -nopad` で再圧縮 |
| 5 | `dd` でランタイムバイナリを切り出し、新 squashfs と結合 |

#### libstdc++ ABI 問題について

AppImage に同梱された `libstdc++.so.6` は Ubuntu 22.04 LTS ベースのビルド環境由来で、
`CXXABI_1.3.13` までしか提供しない。一方 Gentoo (GCC 13+) でビルドした ROCm プロバイダーは
`CXXABI_1.3.15` を要求するため、そのまま差し替えると dlopen 時に以下のエラーが出る:

```
libstdc++.so.6: version `CXXABI_1.3.15' not found
```

スクリプトは `ldconfig -p` でシステムの `libstdc++.so.6` を自動検出し、同梱版と差し替える。
libstdc++ は後方互換なので、エンジン内の他のコンポーネントへの影響はない。

#### MIOPEN_FIND_MODE=FAST について

MIOpen はデフォルト (`NORMAL` モード) で、未キャッシュのコンボリューション設定に対して
全ソルバーを実際に実行・計時して最適なものを選ぶ (Find フェーズ)。
gfx1201 ではテキスト長が変わるたびに入力テンソルサイズが変わるため、
**新しい文言を合成するたびに数秒〜十数秒のスパイク**が生じる。

`MIOPEN_FIND_MODE=FAST` にするとヒューリスティクスで即座にソルバーを選ぶため、
このスパイクが消える。VOICEVOX の合成品質・速度への影響は実測で確認されていない
(平均 RT 比は 16× 前後で変わらず)。

スクリプトは `AppRun` の `apprun=...` 行の直後に `export MIOPEN_FIND_MODE=FAST` を挿入する。

ファイル名を `_cuda.so` のまま維持するのは VOICEVOX core が dlopen する名前に合わせるため
(上記「デプロイ手順」と同じ理由)。

---

## デプロイ手順 (engine 単体配布版への差し替え)

### 1. NVIDIA オリジナルを退避

```bash
cd /path/to/voicevox-engine/linux-nvidia
mkdir -p _nvidia_orig
cp libvoicevox_onnxruntime_providers_cuda.so \
   _nvidia_orig/libvoicevox_onnxruntime_providers_cuda.so.bak
```

### 2. ビルド成果物を差し替え

```bash
cp /path/to/ort-fake-cuda/build/Release/libonnxruntime_providers_rocm.so \
   libvoicevox_onnxruntime_providers_cuda.so
```

> ファイル名が `_cuda.so` のままなのは意図的 (VOICEVOX core が dlopen する名前)。

### 3. NVIDIA 用 stub ライブラリの処理

VOICEVOX 配布物には NVIDIA 用の libcublas, libcudnn, libcurand 等が同梱されていますが、
fake-CUDA EP は内部で ROCm を直接呼ぶのでこれらは**使われません**。残しておいても害は
ありません (provider plugin は `libonnxruntime_providers_cuda.so` 名前で呼ばれるだけで、
中身は ROCm API しか参照していない)。

> ZLUDA 経路を試した際の symlink (libcublas → ~/zluda/... 等) があったら戻しておくこと。
> NVIDIA 配布の original ファイルでも、ROCm EP は import 時に解決を必要としない。

---

## 動作確認

### 起動

```bash
cd /path/to/voicevox-engine/linux-nvidia
HIP_VISIBLE_DEVICES=0 ./run --use_gpu --host 127.0.0.1 --port 50053
```

ログに以下が出れば EP 認識 OK:

```
GPUをテストします:
  * CUDA (device_id=0): OK
CUDA (device_id=0)を利用します
```

### 出力の数値検証 (重要)

「動く」だけでは不十分です。出力が CPU リファレンスと数値的に一致するか確認:

```bash
# CPU リファレンスを別ポートで取得
./run --host 127.0.0.1 --port 50054 &  # GPU なし

# 同じテキストで両方から synthesis を取得して SNR を計算
# (スクリプト例は別途用意 — 16-bit PCM 同士の MSE 比較)
```

正常なら **SNR ≥ +20 dB** (実用上は +80 dB 程度になる)。
0 dB 近辺なら GPU 出力が壊れている → 何か追加の問題があるのでビルド構成・パッチ適用を再確認。

---

## 診断用環境変数 (01-essential 由来)

問題の切り分けに使えます。実機では普段 unset で OK。

| 環境変数 | 効果 |
|---|---|
| `ROCM_DENY_OPS="OpType1,OpType2"` | これらの op を ROCm EP が claim しない (= CPU フォールバック) |
| `ROCM_ALLOW_OPS_ONLY="OpType1,OpType2"` | このリストの op だけ ROCm に置く、他は CPU |
| `ROCM_LOG_OP_PLACEMENT=1` | 各 node の placement 決定を stderr に出力 |

例: 全 op を CPU に逃して fake-CUDA EP のインフラ正常性のみ確認
```bash
ROCM_ALLOW_OPS_ONLY="__none__" ./run --use_gpu ...
```

例: Softmax だけ ROCm で他は CPU
```bash
ROCM_ALLOW_OPS_ONLY="Softmax" ROCM_LOG_OP_PLACEMENT=1 ./run --use_gpu ...
```

このツールは `GPU_WARP_SIZE` バグ特定時 (Softmax を犯人として bisect) に作成しました。

---

## 既知の制限

1. **CPU 比 1/3.5, CUDA RTX 3060 比 2× 程度の時間で動作**。RX 9070 XT のハード
   スペック比からするとまだ最適化余地が大きいが、本パッチでは「正しく動かす」ところまで。
   性能をさらに追求するなら以下が候補:
   - AMD 公式 fork (`https://github.com/ROCm/onnxruntime.git` の
     `rel-1.24.4-rocm7.2` ブランチ) への全面切替 (ORT main lib も差し替え必要)
   - ROCm 7.3+ で MIOpen が gfx1201 用 CK ベース conv solver を wire up する
     のを待つ (本パッチ群は MIOpen 側に依存、ORT 側でできることは限定的)

2. **TunableOp は使えない**。`-fno-rtti` ビルドでは
   `framework/tunable.h:297` でランタイムエラー。RTTI を有効にすると
   `libvoicevox_onnxruntime.so` (RTTI なし) との間で型情報不整合の ABI クラッシュ
   リスクがあるので、TunableOp 活性化は ORT main lib も自前リビルドする覚悟が必要。

3. **CDNA (gfx9xx) 環境では `GPU_WARP_SIZE = 32` のままだと不正確**。
   CDNA wave64 では 64 にする必要あり (`01-essential.patch` 中の該当行をパッチ後に
   再修正、もしくは `01-essential.patch` を CDNA 向けに改変)。

4. **fake-CUDA disguise は CUDA 専用バイナリ全般に応用可能だが、本パッチは
   VOICEVOX が呼び出す機能セットでしか検証していない**。他のアプリ (PyTorch on
   CUDA バイナリ等) で同じ手を使う場合は、その アプリが要求する provider option /
   API 範囲が ROCm EP でカバーされているか個別検証が必要。

---

## ライセンス

本パッチは ONNX Runtime (MIT License) への改変であり、改変版にも MIT License が
適用されます。

## クレジット

検証・パッチ作成: shimada@walbrix.com (with Claude Opus 4.7)

検証日: 2026-06-01 〜 2026-06-03 / VOICEVOX engine 0.25.2 / ROCm 7.2.53210 /
ORT v1.17.3 / RX 9070 XT (gfx1201)
