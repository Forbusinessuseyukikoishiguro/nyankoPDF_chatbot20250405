# 2025/04/03_test_OK_VSCODE版
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
にゃんこ口調のPDF Q&Aシステム
.envファイルからAPIキーを読み込み、指定されたPDFに対して質問を実行するにゃ！
複数の質問を連続して入力できるようになったにゃん♪
"""

import os
import sys
import argparse
from pathlib import Path
import traceback

# .envファイルから環境変数を読み込むライブラリにゃ
from dotenv import load_dotenv


def main():
    # .envファイルを読み込むにゃ
    load_dotenv()

    # コマンドライン引数の設定するにゃん
    parser = argparse.ArgumentParser(description="PDFを元にしたQ&A機能を実行するにゃ！")
    parser.add_argument(
        "--pdf", type=str, help="PDFファイルへのパスを教えてにゃ", required=True
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI APIキー (指定しない場合は.envから取得するにゃ)",
    )
    parser.add_argument(
        "--model", type=str, help="使用するモデル名にゃん", default="gpt-4"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="実行する質問はなんだにゃ？",
        default="このドキュメントの主要ポイントを教えて",
    )
    parser.add_argument(
        "--temperature", type=float, help="生成時の温度パラメータにゃ〜", default=0.0
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="生成する最大トークン数はどれくらいかにゃ？",
        default=None,
    )
    parser.add_argument(
        "--interactive", action="store_true", help="対話モードで実行するかにゃ？"
    )

    args = parser.parse_args()

    # APIキーの設定（優先順位: コマンドライン引数 > .env > 環境変数）にゃん
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    elif "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        print("にゃ〜ん！OpenAI APIキーが設定されていないにゃ...")
        print("以下のいずれかの方法でAPIキーを設定してほしいにゃ:")
        print("1. --api_key 引数で指定するにゃ")
        print("2. .envファイルにOPENAI_API_KEY=sk-あなたのキー を記述するにゃ")
        print("3. OPENAI_API_KEY 環境変数を設定するにゃん")
        return 1

    # APIキーが設定されていることの確認にゃ
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not api_key.startswith("sk-"):
        print("にゃにゃ？APIキーの形式が正しくない可能性があるにゃ...")
        print(f"設定されたAPIキーの先頭: {api_key[:5]}...")
    else:
        print(
            f"APIキー確認したにゃん！ちゃんと設定されてるにゃ (先頭: {api_key[:5]}...)"
        )

    # PDFファイルのパスにゃん
    pdf_path = Path(args.pdf)

    # PDFファイルが存在するか確認するにゃ
    if not pdf_path.exists():
        print(f"にゃぁ〜！指定されたPDFファイル '{pdf_path}' が見つからないにゃ...")
        return 1

    try:
        # 最新バージョンのllama-indexを使用するにゃん
        print("最新のllama-indexを使うにゃん...")

        # 必要なライブラリのインポートにゃ - 互換性対応版
        try:
            # 新しいバージョンを試すにゃん
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.core.readers import SimpleDirectoryReader
            from llama_index.llms.openai import OpenAI
            from llama_index.core import StorageContext

            # 様々なバージョンに対応するにゃん
            try:
                # 最新バージョンのロード方法
                load_index_func = VectorStoreIndex.from_storage
            except AttributeError:
                try:
                    # 中間バージョンのロード方法
                    from llama_index.core import load_index_from_storage

                    load_index_func = load_index_from_storage
                except ImportError:
                    # 古いバージョンのロード方法
                    from llama_index import load_index_from_storage

                    load_index_func = load_index_from_storage

            using_new_version = True
        except ImportError:
            # 古いバージョンを試すにゃん
            from llama_index import (
                VectorStoreIndex,
                ServiceContext,
                load_index_from_storage,
            )
            from llama_index import SimpleDirectoryReader
            from llama_index.llms.openai import OpenAI
            from llama_index import StorageContext

            load_index_func = load_index_from_storage
            using_new_version = False

        # PDFリーダーを用意するにゃん
        reader = SimpleDirectoryReader(input_files=[str(pdf_path)])

        # PDFの読み込みにゃん
        print(f"PDFファイル '{pdf_path}' を読み込み中...にゃん")
        documents = reader.load_data()
        print(
            f"PDFの読み込み完了したにゃ！ {len(documents)} ドキュメントに分割したにゃ"
        )

        # 指定されたモデルを使用するにゃー
        if using_new_version:
            # 新しいバージョン用の設定にゃん
            llm = OpenAI(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            # 最新の設定インターフェースを使用にゃん
            Settings.llm = llm
            Settings.chunk_size = 1024
        else:
            # 古いバージョン用の設定にゃん
            llm = OpenAI(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            service_context = ServiceContext.from_defaults(llm=llm)

        print(f"llama-index を使ってるにゃ (モデル: {args.model})")

        # インデックスの作成とキャッシュディレクトリにゃん
        storage_dir = f"./storage_{pdf_path.stem}_{args.model.replace('-', '_')}/"

        # キャッシュが存在するか確認するにゃん
        has_cache = Path(storage_dir).exists()

        if has_cache:
            print(f"既存のインデックスを見つけたにゃん: {storage_dir}")
            print("保存済みインデックスを読み込むにゃ...")

            # インデックスを読み込むにゃん - 互換性対応版
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            try:
                index = load_index_func(storage_context)
            except Exception as e:
                print(f"インデックス読み込みでエラーが起きたにゃ: {e}")
                print("インデックスを再作成するにゃん...")
                has_cache = False
                # 古いキャッシュが問題を起こしているようなら削除するにゃん
                import shutil

                try:
                    shutil.rmtree(storage_dir)
                    print(f"古いインデックスを削除したにゃん: {storage_dir}")
                except:
                    print("古いインデックスの削除に失敗したにゃ...")

        # キャッシュがないか読み込みに失敗した場合は新規作成するにゃん
        if not has_cache:
            print("インデックスが見つからないから新しく作るにゃん...")
            print("ベクトルインデックスを作成中...にゃん")

            if using_new_version:
                # 新しいバージョン用
                index = VectorStoreIndex.from_documents(documents)
            else:
                # 古いバージョン用
                index = VectorStoreIndex.from_documents(
                    documents, service_context=service_context
                )

            # インデックスを保存するにゃん
            os.makedirs(storage_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=storage_dir)
            print(f"インデックスを '{storage_dir}' に保存したにゃん！")

        # クエリエンジンの作成にゃん
        print("クエリエンジンを初期化中...にゃん")
        query_engine = index.as_query_engine()

        # 質問を実行する関数を定義するにゃん
        def execute_query(query_text):
            print(f"\n=== 質問: {query_text} ===")
            print("考え中...にゃん...")
            response = query_engine.query(query_text)
            print(f"\n回答にゃん！\n{response}")

            # ソース表示を試みるにゃ
            print("\n=== 参照元にゃん: ===")
            try:
                if hasattr(response, "source_nodes") and response.source_nodes:
                    for i, source in enumerate(response.source_nodes):
                        print(f"参照元 {i+1} にゃん:")
                        if hasattr(source.node, "get_content"):
                            print(source.node.get_content()[:300] + "...")
                        elif hasattr(source.node, "get_text"):
                            print(source.node.get_text()[:300] + "...")
                        else:
                            print("参照元の内容を表示できないにゃ...")
                else:
                    print("参照元情報が見つからないにゃ...")
            except Exception as e:
                print(f"参照元の表示中にエラーが起きたにゃ: {e}")

            return response

        # 最初の質問を実行するにゃん
        execute_query(args.query)

        # 質問後に次の質問があるか確認するにゃん
        def ask_for_more_questions():
            while True:
                next_step = input(
                    "\nまだ質問ある？ ('はい'で続行、'おわり'で終了するにゃ) > "
                )
                if next_step.lower() in ["おわり", "終了", "exit", "quit", "no", "n"]:
                    print("さよにゃら〜！またね！")
                    return False
                elif next_step.lower() in ["はい", "続ける", "yes", "y", "続行"]:
                    return True
                else:
                    print("'はい' か 'おわり' で答えてほしいにゃ～")

        # 対話モードか引数で指定した場合は複数の質問を受け付けるにゃん
        if args.interactive:
            print("\n=== にゃんこ先生の対話モードにゃん！===")
            print("質問を入力するか、'おわり'と入力して終了するにゃ\n")

            while True:
                has_more = ask_for_more_questions()
                if not has_more:
                    break

                user_query = input("\nにゃんこ先生に質問するにゃ？> ")
                if user_query.lower() in ["おわり", "終了", "exit", "quit"]:
                    print("さよにゃら〜！またね！")
                    break
                if not user_query.strip():
                    print("何か質問を入力するにゃ！")
                    continue

                execute_query(user_query)
        else:
            # 対話モードでなくても最初の質問の後に続けるか聞くにゃん
            while True:
                has_more = ask_for_more_questions()
                if not has_more:
                    break

                user_query = input("\nにゃんこ先生に質問するにゃ？> ")
                if user_query.lower() in ["おわり", "終了", "exit", "quit"]:
                    print("さよにゃら〜！またね！")
                    break
                if not user_query.strip():
                    print("何か質問を入力するにゃ！")
                    continue

                execute_query(user_query)

        print("\n処理が完了したにゃん！また質問してにゃ♪")
        return 0

    except Exception as e:
        print(f"にゃにゃにゃ！エラーが発生したにゃん: {e}")
        traceback.print_exc()

        print("\n問題解決のヒントにゃん:")
        print("1. 必要なライブラリをインストールしてほしいにゃ:")
        print("   pip install -U python-dotenv llama-index openai pypdf langchain")
        print("2. APIキーが正しく設定されているか確認するにゃ (.envファイル)")
        print("3. PDFファイルが正しく配置され、アクセスできるか確認するにゃん")
        print("4. モデル名が正しいか確認するにゃ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
