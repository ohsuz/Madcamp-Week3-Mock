package com.app.madcampweek3;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import com.app.madcampweek3.ui.note.NoteFragment;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;

public class QandA_Activity extends AppCompatActivity {
    String mock_id="";
    RecyclerView qandaView;
    ArrayList<QandAItem> qandaList;
    QandAAdapter qandaAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_qanda);

        Intent intent = getIntent();
        mock_id = intent.getStringExtra("mock_id");

        qandaView = findViewById(R.id.qandaView);
        qandaView.setHasFixedSize(true);
        qandaView.setLayoutManager(new LinearLayoutManager(getApplicationContext()));

        qandaList = new ArrayList<>();

        qandaAdapter = new QandAAdapter(qandaList);
        qandaView.setAdapter(qandaAdapter);

        final  GetQandA getQandA = new GetQandA();
        getQandA.start();
        try {
            getQandA.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        qandaView.addOnItemTouchListener(new RecyclerTouchListener(getApplicationContext(), qandaView, new RecyclerTouchListener.ClickListener() {
            @Override
            public void onClick(View view, int position) {
            }

            @Override
            public void onLongClick(View view, int position) { }
        }));

    }

    public class GetQandA extends Thread {

        @Override
        public void run() {
            String serverUri = "http://ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/getWrong.php";
            String parameters = "user=" + User.email+"&mock_id="+mock_id;
            try {
                URL url = new URL(serverUri);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoInput(true);
                connection.setUseCaches(false);

                OutputStream outputStream = connection.getOutputStream();
                outputStream.write(parameters.getBytes("UTF-8"));
                outputStream.flush();
                outputStream.close();

                InputStream is = connection.getInputStream();
                InputStreamReader isr = new InputStreamReader(is);
                BufferedReader reader = new BufferedReader(isr);

                final StringBuffer buffer = new StringBuffer();
                String line = reader.readLine();

                while (line != null) {
                    buffer.append(line + "\n");
                    line = reader.readLine();
                }

                //읽어온 문자열에서 row(레코드)별로 분리하여 배열로 리턴하기
                String[] rows=buffer.toString().split(";");

                for(String row : rows){
                    //한줄 데이터에서 한 칸씩 분리
                    String[] datas=row.split("&");
                    if(datas.length!=4) continue;
                    String qanda = "[ 문제 "+datas[0]+"번 ] "+datas[1];
                    String q_image= "http://ubuntu@ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/"+datas[2];
                    String explanation="http://ubuntu@ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/"+datas[3];

                    QandAItem qandaItem = new QandAItem(qanda, q_image, explanation);
                    qandaList.add(qandaItem);
                    qandaAdapter.notifyDataSetChanged();
                }
                qandaView.setAdapter(qandaAdapter);
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}