Decode with Python API
======================

Once ELIT is `installed <install.html>`_, NLP components can be used to decode raw text into linguistic structures.


Create NLP Pipeline
-------------------

The followings show how to create an NLP pipeline with 6 core components:

.. code:: python

   from elit.component import EnglishTokenizer
   from elit.component import EnglishMorphAnalyzer
   from elit.component import POSFlairTagger
   from elit.component import NERFlairTagger
   from elit.component import DEPBiaffineParser
   from elit.component import SDPBiaffineParser

   tok = EnglishTokenizer()
   morph = EnglishMorphAnalyzer()
   pos = POSFlairTagger()
   ner = NERFlairTagger()
   dep = DEPBiaffineParser()
   sdp = SDPBiaffineParser()

   pipeline = [tok, morph, pos, ner, dep, sdp]

Take a look at the individual tool page for more details about available components:

* `Tokenizer <../tools/tokenization.html>`_
* `Morphological Analyser <../tools/morphological_analysis.html>`_
* `Part-Of-Speech Tagger <../tools/part_of_speech_tagging.html>`_
* `Named Entity Recognizer <../tools/named_entity_recognition.html>`_
* `Dependency Parser <../tools/dependency_parsing.html>`_
* `Semantic Dependency Parser <../tools/semantic_dependency_parsing.html>`_


Import Models
-------------------

All `pre-trained models <../documentation/models.html>`_ are publicly available in the `ELIT's S3 bucket <http://elit-models.s3.amazonaws.com>`_.
The followings show how to import models for ``pos``, ``ner``, ``dep``, and ``sdp``:

.. code:: python

   from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED
   from elit.resources.pre_trained_models import ELIT_NER_FLAIR_EN_ONTONOTES
   from elit.resources.pre_trained_models import ELIT_DEP_BIAFFINE_EN_MIXED
   from elit.resources.pre_trained_models import ELIT_SDP_BIAFFINE_EN_MIXED

   pos.load(ELIT_POS_FLAIR_EN_MIXED)
   ner.load(ELIT_NER_FLAIR_EN_ONTONOTES)
   dep.load(ELIT_DEP_BIAFFINE_EN_MIXED)
   sdp.load(ELIT_SDP_BIAFFINE_EN_MIXED)

The ``load`` function supports paths from both local and remote directories.


