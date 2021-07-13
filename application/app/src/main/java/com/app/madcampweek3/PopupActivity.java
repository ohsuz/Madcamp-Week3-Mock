package com.app.madcampweek3;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class PopupActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_popup);

        Intent intent = getIntent();
        String totalNum = Integer.toString(intent.getIntExtra("total", 0));
        String wrongNum =  Integer.toString(intent.getIntExtra("wrong", 0));
        final String mock_id = intent.getStringExtra("mock_id");

        TextView result = findViewById(R.id.result);
        Button okayBtn = findViewById(R.id.okayBtn);
        Button noBtn = findViewById(R.id.noBtn);

        String resultString = "총 "+totalNum+"문제 중 "+wrongNum+"문제를 틀렸습니다.\n바로 오답노트로 이동할까요?";

        result.setText(resultString);

        okayBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(PopupActivity.this, QandA_Activity.class);
                intent.putExtra("mock_id", mock_id);
                startActivity(intent);
                finish();
            }
        });

        noBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });


    }
}