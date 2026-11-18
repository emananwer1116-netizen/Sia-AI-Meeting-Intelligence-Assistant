# Sia — AI Meeting Assistant (MVP محلي)

Sia بيحوّل تسجيل اجتماع (Mentor + طلاب) إلى نص مقسّم حسب المتحدثين (ElevenLabs)، وبعد ما **انت** تحدد الأسماء والـ Mentor، بيحلل الاجتماع عن طريق Sprints LiteLLM ويطلّع المطلوب من كل شخص **مع الدليل من كلام الاجتماع**. أي مهمة أو قرار ملوش دليل حقيقي في الـ Transcript بيترفض تلقائيًا.

## 1) التثبيت على Windows (PowerShell)

```powershell
# داخل فولدر المشروع Sia
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# لو ظهر خطأ Execution Policy:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

## 2) إعداد ملف `.env`

| المتغير | الوصف |
|---|---|
| `ELEVENLABS_API_KEY` | مفتاح ElevenLabs بتاعك |
| `ELEVENLABS_STT_MODEL` | موديل التفريغ (الافتراضي `scribe_v2`) |
| `LITELLM_BASE_URL` | `https://management.sprints.ai/litellm/` |
| `LITELLM_API_KEY` | مفتاح Sprints LiteLLM |
| `LITELLM_MODEL` | **Model ID الصحيح — فاضي لحد ما تعرفه** |
| `LITELLM_JSON_MODE` | `true` فقط لو الموديل بيدعم `response_format=json_object` |

**إزاي تعرف الـ Model ID؟** شغّل التطبيق، افتح شاشة *4. Meeting Analysis*، افتح *LiteLLM connection helper* واضغط **List available models**، وانسخ الاسم المناسب في `LITELLM_MODEL` (ثم أعد تشغيل التطبيق). لا تضع المفاتيح في الكود أو GitHub: `.env` موجود في `.gitignore`.

## 3) تشغيل الاختبارات

```powershell
python -m pytest
```
الاختبارات كلها بتستخدم Mock، مش بتتصل بـ ElevenLabs ولا LiteLLM ومش بتستهلك رصيد.

## 4) تشغيل التطبيق

```powershell
streamlit run app.py
```
افتح الرابط اللي بيظهر (غالبًا `http://localhost:8501`).

## 5) طريقة الاستخدام

1. **Upload Meeting**: ارفع الملف، راجع الاسم والحجم والمدة، واضغط *Save recording* (حفظ محلي فقط). بعدها أكّد أن الطلب هيستهلك رصيد ElevenLabs واضغط *Start transcription*.
2. **Transcript**: عرض للقراءة فقط، مقسّم حسب المتحدث مع التوقيتات.
3. **Speaker Identification**: اكتب الاسم والدور لكل Speaker، واختر الـ Mentor بنفسك (مفيش تخمين). الحقل *Other spellings* مفيد لو الاسم بالإنجليزي والاجتماع بالعربي (مثلًا `Sara` ← `سارة`) عشان Sia يتأكد إن الاسم اتقال فعلًا.
4. **Meeting Analysis**: اضغط *Analyze Meeting*. الطلب بيتبعت مرة واحدة حتى لو ضغطت أكتر من مرة.
5. **Results**: ملخص، قرارات، أفكار، مهام (قابلة للتعديل)، عرض حسب الشخص، اعتماديات، عوائق، أسئلة مفتوحة، توجيهات الـ Mentor، ومراجعة ما رفضه المدقّق. اضغط *Save changes* لحفظ تعديلاتك (النسخة الأصلية بتفضل محفوظة).
6. **Save / Open Meetings**: كل الاجتماعات محفوظة محليًا في `storage/`. فتح اجتماع قديم **لا** يستهلك ElevenLabs.

## 6) حماية الرصيد والبيانات

- التسجيل لا يُرسل لـ ElevenLabs إلا بعد تأكيدك الصريح، ولا يُرسل مرة تانية إلا بموافقة ثانية.
- لا يوجد Retry تلقائي لأي طلب (ElevenLabs أو LiteLLM).
- الاستجابة الخام من ElevenLabs بتتحفظ فورًا (`storage/transcripts/<id>.raw.json`) قبل أي معالجة، ولو حصل خطأ في المعالجة زرار *Rebuild from saved response* بيصلحها **مجانًا**.
- نفس التسجيل (نفس الـ SHA-256) ما بيتضافش مرتين.
- لا يتم حذف أي تسجيل أو Transcript أو تقرير؛ النسخ القديمة عند إعادة الإرسال/التحليل بتتأرشف بتاريخ.
- الحدود الافتراضية: 500 MB و240 دقيقة (قابلة للتغيير في `.env`). وثائق ElevenLabs الحالية تذكر حد أقصى للملف أقل من 5 GB.

## 7) قواعد فهم الاجتماع (الحماية من الاختراع)

كل عنصر في التقرير لازم يكون مربوط بـ `segment_id` وجملة اقتباس **موجودة فعلًا** في نفس الـ segment (بعد توحيد التشكيل والهمزات). ثم:

- **بدون دليل مؤكد** ← العنصر يتشال من التقرير ويظهر في *Validator → Rejected*.
- **المسؤول** لازم يكون مذكور بالاسم في جملة الـ Mentor (أو الالتزام صادر من نفس الشخص)، وإلا الحالة تبقى *Needs Review*. مهمة بدون مسؤول واضح = *Unassigned*.
- **الموعد** لازم يكون موجود حرفيًا في الدليل، وإلا بيتشال. Sia لا يحوّل "الخميس" إلى تاريخ.
- **قرار** مدعوم بمتحدث واحد غير الـ Mentor فقط ← *Needs Review* (ممكن يكون اقتراح).
- التعليمات المتعارضة بتتعرض كـ Conflicts ولا يتم اختيار واحدة.

> هذه الضمانات تضبط **مخرجات** النموذج. جودة الفهم نفسها (التمييز بين اقتراح وقرار في لهجة مصرية حقيقية) تعتمد على الموديل اللي هتختاره — جرّبها على اجتماع حقيقي وراجع شاشة Results.

## 8) حل أشهر الأخطاء

| المشكلة | الحل |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `streamlit` غير معروف | فعّل الـ venv ثم `pip install -r requirements.txt` |
| `Missing in .env: ELEVENLABS_API_KEY` | تأكد أن `.env` (مش `.env.example`) جنب `app.py` وأعد تشغيل التطبيق |
| `401` من ElevenLabs/LiteLLM | المفتاح غلط أو منتهي |
| `404` من LiteLLM | راجع `LITELLM_MODEL` (استخدم List available models) |
| `400` ومذكور `response_format` | خلّي `LITELLM_JSON_MODE=false` |
| `finish_reason=length` | موديل بحد مخرجات أكبر أو `LITELLM_MAX_TOKENS` |
| `429` | حد معدّل/رصيد؛ استنى وأعد الإرسال يدويًا |
| الملف أكبر من الحد | غيّر `SIA_MAX_AUDIO_MB` / `SIA_MAX_AUDIO_MINUTES` (و`maxUploadSize` في `.streamlit/config.toml`) |
| Transcript طويل جدًا للتحليل | الـ MVP يحلل في طلب واحد (حد `SIA_MAX_PROMPT_CHARS`)؛ التقسيم لأجزاء ليس ضمن هذه النسخة |

## 9) هيكل المشروع

```
app.py                      واجهة Streamlit فقط
config/settings.py          الإعدادات من .env
services/                   elevenlabs_client, transcription_service, litellm_client,
                            meeting_analyzer, report_service, audio_utils, jobs, text_utils
schemas/                    transcript.py, meeting_report.py (Pydantic)
database/db.py              SQLite
storage/                    recordings / transcripts / reports (محلي، خارج Git)
tests/                      98 اختبار + بيانات اجتماع مصري وهمي
```
