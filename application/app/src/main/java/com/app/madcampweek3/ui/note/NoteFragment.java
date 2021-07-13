package com.app.madcampweek3.ui.note;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;
import androidx.lifecycle.ViewModelProviders;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.app.madcampweek3.NoteAdapter;
import com.app.madcampweek3.NoteItem;
import com.app.madcampweek3.QandA_Activity;
import com.app.madcampweek3.R;
import com.app.madcampweek3.RecyclerTouchListener;
import com.app.madcampweek3.User;
import com.app.madcampweek3.ui.capture.CaptureFragment;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;

public class NoteFragment extends Fragment {

    RecyclerView noteView;
    ArrayList<NoteItem> noteList;
    NoteAdapter noteAdapter;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.fragment_note, container, false);

        noteView = root.findViewById(R.id.noteView);
        noteView.setHasFixedSize(true);
        noteView.setLayoutManager(new LinearLayoutManager(getContext()));

        noteList = new ArrayList<>();

        noteAdapter = new NoteAdapter(noteList);
        noteView.setAdapter(noteAdapter);

        final GetNotes getNotes = new GetNotes();
        getNotes.start();
        try {
            getNotes.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        noteView.addOnItemTouchListener(new RecyclerTouchListener(getContext(), noteView, new RecyclerTouchListener.ClickListener() {
            @Override
            public void onClick(View view, int position) {
                String mock_id = noteList.get(position).getMock_id();
                Intent intent = new Intent(getContext(), QandA_Activity.class);
                intent.putExtra("mock_id", mock_id);
                startActivity(intent);
            }

            @Override
            public void onLongClick(View view, int position) { }
        }));

        return root;
    }

    public class GetNotes extends Thread {

        @Override
        public void run() {
            String serverUri = "http://ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/getNotes.php";
            String parameters = "user=" + User.email;
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
                    if(datas.length!=5) continue;
                    String mock_id = datas[0];
                    String mock_name="";
                    String mock_subject=datas[4];
                    if(!datas[3].equals("수능")) {
                        mock_name = "고" + datas[1] + " " + datas[2] + "년도 " + datas[3] + "월\n" + datas[4] + " 모의고사";
                    }else{
                        mock_name = "고" + datas[1] + " " + datas[2] + "년도 " + datas[3] + "\n" + datas[4] + " 모의고사";
                    }
                    NoteItem noteItem = new NoteItem(mock_id, mock_name, mock_subject);
                    noteList.add(noteItem);
                    noteAdapter.notifyDataSetChanged();
                }
                noteView.setAdapter(noteAdapter);
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}