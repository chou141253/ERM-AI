# 雙擊發布程式與模型

在 Windows 檔案總管進入 `new/`，雙擊 **`publish_erm_ai.bat`**。它依序完成 GitHub 與 Hugging Face 兩個流程。

## GitHub：自動 fork，再推到你的分支

1. 先執行本機測試，檢查要公開的檔案。
2. 若電腦未安裝 GitHub CLI，下載官方可攜版到 `.publish-work/` 並核對 SHA256；不需要更動系統 Git。
3. 第一次透過瀏覽器登入 **你的個人 GitHub 帳號**。
4. 自動建立或重用 `<登入帳號>/ERM-AI` fork，確認它來自 `ntuhophai/ERM-AI`。同名無關 repo 會讓流程停止。
5. 把新版程式、設定、README 和測試提交到 fork 的新分支，原 repo 和 fork 的預設分支均不改動。
6. 成功後開啟 GitHub 比較頁，由你按 **Create pull request**。程式不替你建立或合併 PR。

啟動檔以本機實際登入的帳號為準，不硬編碼 `chou141253`。若登入的是原 repo 擁有者 `ntuhophai`，無法 fork 自己的 repo；請登入你想建立 fork 的個人帳號。

公開 repo 保持 `new/` 程式目錄結構，另建立根目錄 README。`old/`、資料、模型、病人 manifests、預測結果、環境與工作暫存都不包含在 source 提交中。只處理明確的程式／文件清單，不執行 `git add .`。

## Hugging Face：直接推送你選的模型

1. 選擇視窗讓你選 **一個 `.pth`／`.pt` checkpoint**。
2. 新版 checkpoint 自帶架構、stream 順序、標籤與前處理，會直接核對。
3. 若是 `old/` 的原始權重，另外選對應 YAML。八流可選 `publish_profiles/legacy_8_stream.yaml`；其他模型需準備正確設定，不能套用八流設定。
4. 預覽顯示公開設定。原始權重不能證明 stream 的語意順序，因此 legacy 模型需輸入 `YES`，確認它符合原實驗。
5. 匯出純權重 `model.safetensors`，metadata 使用允許清單，不帶病人／病例編號、完整訓練設定、本機路徑或內部切分指紋。
6. 第一次使用官方 Hugging Face 本機登入提示。請用對 **`ntuhophai/ERM-AI` 有寫入權限**的帳號；可走瀏覽器授權，或在本機提示輸入適當寫入權限的 Token。不要把 Token 寫進 BAT、Python、README 或聊天。
7. 將以下五個檔案直接 commit 到 `https://huggingface.co/ntuhophai/ERM-AI`，不建立 Hugging Face PR：

```text
model.safetensors
config.json
modeling_multistream.py
README.md
requirements.txt
```

**這會直接更新模型 repo 根目錄的同名檔案。** 新 commit 保留既有其他檔案。每次只選一個 checkpoint；重複上傳到根目錄不能保存多個 fold。

若模型或 legacy 設定視窗按取消，模型階段會略過，已成功的 GitHub 提交仍保留。

### 舊模型的前處理

八流 profile 依現存設定及 2025-09-23 W&B 環境描述：ResNet50、448 crop、tensor resize、`antialias: false`。其他實驗的 torchvision 版本可能不同；例如 2025-09-18 雙流環境為 0.17.1，resize 的預設 antialias 已變為 true。因此 legacy 設定必須明確寫 `resize_backend` 與 `antialias`。

鍵與尺寸檢查只能確認架構，不能證明歷史 stream 順序、標籤臨床意義或評估獨立性。Model Card 保留這些限制。`smoke_output/` 模型會標示 demo，不能作為正式眼科訓練模型。

## 其他啟動檔

| 檔案 | 動作 |
|---|---|
| `publish_erm_ai.bat` | GitHub fork／分支提交，接著 Hugging Face 模型直接 push |
| `publish_github.bat` | 只做 GitHub |
| `publish_huggingface.bat` | 只做 Hugging Face，適合登入／網路問題後重試 |
| `publish_preview.bat` | 只列出公開程式清單；不登入、不 fork、不上傳 |

每次 GitHub 發布建立新分支，不 force push。模型錯誤不撤回已完成的 GitHub 提交。視窗最後停留，方便閱讀成功訊息或錯誤。

## 環境與暫存

優先使用 `new/.venv/`。若不存在，嘗試用已安裝的 Python 3.10 以上在 `.publish-work/runtime/` 建立隔離環境；首次安裝 PyTorch 等依賴可能需要幾分鐘與較多磁碟空間。Python 太舊或未安裝時會明確停止。

`.publish-work/` 保存程式清單、公開模型包與遠端 commit 資訊，方便重試；它不會提交到 GitHub。本機原始權重與舊實驗不會刪除或搬動。憑證由官方工具保存在本機，腳本不讀出或公開 Token。

只有按啟動檔才會建立 fork／提交／上傳。可先用 `publish_preview.bat`。交付前的檢查只做本機語法、測試與匯出驗證，沒有替你執行遠端發布。

## 本機驗證

2026-10-12 完整測試為 **79 passed、16 subtests passed**。已實際執行 Preview BAT（含中文目錄）、PowerShell 語法檢查、Python 預檢，以及官方 GitHub CLI 可攜版 checksum／版本核對。原有八流 fold 0 權重通過完整鍵／尺寸預覽；合成單／多流模型匯出後載回，預測 logits 相同。

GitHub fork／blob／commit／分支及 Hugging Face 上傳測試使用模擬 API，沒有實際建立 fork、PR 或執行 push。因此本機流程已驗證，但真正帳號權限與遠端傳輸仍要在你執行啟動檔時確認。

## 參考

- [GitHub CLI fork](https://cli.github.com/manual/gh_repo_fork)
- [GitHub CLI 登入](https://cli.github.com/manual/gh_auth_login)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Hugging Face 上傳](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Torchvision Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html)
