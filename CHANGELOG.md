# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased] 
### Added
### Changed
### Removed

## [0.1.21] - 2018-7-17
### Added
- Add mxnet to deps
### Changed
- Inherited component from elit sdk
- Flatten structure
- Update POS tagger
### Removed

## [0.1.20] - 2018-6-18
### Added
- Segmenter test
### Changed
- Fix Segmenter
- Seperate Segmenter out form Tokenizer
### Removed

## [0.1.19] - 2018-06-15
### Added
### Changed
- Add filed name: tok and offset to SpaceTokenizer and EnglishTokenizer
- Update tokenizer tests
### Removed

## [0.1.18] - 2018-06-10
### Added
### Changed
- Change result format. 'token' -> 'tok'. Add 'sid' field
- Update elitsdk to 0.0.6
### Removed

## [0.1.17] - 2018-05-23
### Added
### Changed
- Refactor tokenizer
- Refartor tokenizer test
### Removed

## [0.1.16] - 2018-5-21
### Added
### Changed
- Replace fasttext with yafasttext
### Removed

## [0.1.15] - 2018-03-15
### Added
- Deep Biaffine Attention Neural Dependency Paser by @hankcs
### Changed
- NER
- NLPState and NLPModel
- Lexicon
    - Add marisa_trie into dependency
- Configuration: sentiment model is not loaded by default
### Removed

## [0.1.14] - 2017-10-29
### Added
- Sentiment Analysis
- It is first released.


[Unreleased]: https://github.com/elitcloud/elit/compare/0.1.6...HEAD
[0.1.5]: https://github.com/elitcloud/elit/compare/0.1.5...0.1.6
[0.1.4]: https://github.com/elitcloud/elit/compare/0.1.4...0.1.5